"""Single-equation views and 2SLS sensitivity using the shared within estimator.

Inputs: matched finite arrays. Outputs: two-way inference and IV relevance.
AAC exclusion remains unverified; IV is a sensitivity, never the main pricing input.
"""

import numpy as np
import pandas as pd
from scipy import stats
from .econometrics import (
    fit_system,
    absorb as within_projection,
    fixed_effect_rank,
    score_covariance,
)


def absorb(values, groups):
    """Project values off the store/week groups.
    Return residualized values, an unused scalar placeholder and maximum group-mean error.
    """
    transformed = within_projection(values, groups[0], groups[1])
    error = max(
        np.max(
            np.abs(pd.DataFrame(transformed).groupby(np.asarray(g)).mean().to_numpy())
        )
        for g in groups
    )
    return transformed, np.nan, error


def two_way_covariance(x_scores, residuals, bread, store, week, degrees_of_freedom):
    """Combine x_scores and residuals with bread and store/week cluster labels.
    Return the finite-sample-corrected store-plus-week-minus-intersection covariance."""
    scores = x_scores * residuals[:, None]
    n = len(scores)
    parameters = n - degrees_of_freedom
    intersection = np.unique(
        np.column_stack([store, week]), axis=0, return_inverse=True
    )[1]
    covariance = (
        score_covariance(scores, bread, store, n, parameters)
        + score_covariance(scores, bread, week, n, parameters)
        - score_covariance(scores, bread, intersection, n, parameters)
    )
    return (covariance + covariance.T) / 2


def fit_ols(y, x, names, store, week):
    """Expose one shared system fit as a named equation for auxiliary exercises."""
    assert np.isfinite(y).all() and np.isfinite(x).all()
    transformed = within_projection(x, store, week)
    retained = np.linalg.norm(transformed, axis=0) > 1e-8
    dropped = [name for name, keep in zip(names, retained) if not keep]
    if any(name.startswith("price_") for name in dropped):
        raise ValueError(f"No identifying price variation: {dropped}")
    used_names = [name for name, keep in zip(names, retained) if keep]
    frame = pd.DataFrame(np.asarray(x)[:, retained], columns=used_names)
    frame["outcome"], frame["store"], frame["week"] = (
        np.asarray(y),
        np.asarray(store),
        np.asarray(week),
    )
    fit = fit_system(frame, ["outcome"], used_names, np.ones(len(frame), bool), True)
    yw = within_projection(np.asarray(y)[:, None], store, week)[:, 0]
    return dict(
        coefficients=fit.beta[:, 0],
        covariance=fit.covariance,
        names=used_names,
        dropped=dropped,
        residuals=fit.residual[:, 0],
        fitted=fit.fitted[:, 0],
        x_within=fit.within_x,
        y_within=yw,
        bread=fit.bread,
        store=np.asarray(store),
        week=np.asarray(week),
        n=fit.n,
        df=fit.n - fit.parameters,
        stores=fit.stores,
        weeks=fit.weeks,
        iterations=np.nan,
        absorption_error=float(np.max(np.abs(fit.within_x.T @ fit.residual))),
        condition_number=float(np.linalg.cond(fit.within_x)),
        within_r_squared=float(1 - fit.residual[:, 0] @ fit.residual[:, 0] / (yw @ yw)),
    )


def coefficient_rows(fit, category, model, outcome):
    """Return long-form coefficients and t intervals with the smaller cluster df."""
    se = np.sqrt(np.maximum(np.diag(fit["covariance"]), 0))
    df = min(fit["stores"], fit["weeks"]) - 1
    critical = stats.t.ppf(0.975, df)
    rows = []
    for index, name in enumerate(fit["names"]):
        coefficient = float(fit["coefficients"][index])
        rows.append(
            dict(
                category=category,
                model=model,
                outcome=outcome,
                term=name,
                coefficient=coefficient,
                standard_error=float(se[index]),
                ci_low=coefficient - critical * se[index],
                ci_high=coefficient + critical * se[index],
                p_value=(
                    2 * stats.t.sf(abs(coefficient / se[index]), df)
                    if se[index] > 0
                    else np.nan
                ),
                n=fit["n"],
                stores=fit["stores"],
                weeks=fit["weeks"],
                within_r_squared=fit["within_r_squared"],
            )
        )
    return rows


def fit_iv(y, price, controls, instruments, names, store, week):
    """Two-stage least squares using four log AACs for four endogenous log prices.

    Inputs are matched, finite arrays. Return the demand fit and separate
    first-stage relevance diagnostics. Exclusion is an assumption, not a test.
    """
    all_data = np.column_stack([y, price, controls, instruments])
    within, iterations, error = absorb(all_data, [store, week])
    y = within[:, 0]
    x = within[:, 1:9]
    z = np.column_stack([within[:, 9:13], within[:, 5:9]])
    assert np.linalg.matrix_rank(z) == z.shape[1]
    projection = np.linalg.solve(z.T @ z, z.T @ x)
    xhat = z @ projection
    bread = np.linalg.inv(xhat.T @ x)
    beta = bread @ xhat.T @ y
    residuals = y - x @ beta
    degrees = len(y) - x.shape[1] - fixed_effect_rank(store, week)
    covariance = two_way_covariance(xhat, residuals, bread, store, week, degrees)
    if (np.diag(covariance) < -1e-10).any():
        raise ValueError("Negative IV cluster variance")
    fit = dict(
        coefficients=beta,
        covariance=covariance,
        names=names,
        n=len(y),
        stores=len(np.unique(store)),
        weeks=len(np.unique(week)),
        within_r_squared=float(1 - residuals @ residuals / (y @ y)),
        condition_number=float(np.linalg.cond(xhat.T @ x)),
        iterations=iterations,
        absorption_error=error,
        residuals=residuals,
        dropped=[],
    )
    diagnostics = []
    for column in range(4):
        first_beta = np.linalg.lstsq(z, x[:, column], rcond=None)[0]
        first_residual = x[:, column] - z @ first_beta
        first_bread = np.linalg.inv(z.T @ z)
        first_cov = two_way_covariance(
            z, first_residual, first_bread, store, week, degrees
        )
        excluded_cov = first_cov[:4, :4]
        eigenvalue = np.linalg.eigvalsh(excluded_cov).min()
        wald_f = (
            first_beta[:4] @ np.linalg.solve(excluded_cov, first_beta[:4]) / 4
            if eigenvalue > 0
            else np.nan
        )
        restricted = (
            x[:, column]
            - z[:, 4:] @ np.linalg.lstsq(z[:, 4:], x[:, column], rcond=None)[0]
        )
        diagnostics.append(
            dict(
                term=names[column],
                joint_excluded_wald_F=wald_f,
                partial_r_squared=1
                - first_residual @ first_residual / (restricted @ restricted),
                excluded_cov_min_eigenvalue=eigenvalue,
                interpretation="Relevance diagnostic; not a conditional weak-IV statistic or exclusion test",
            )
        )
    return fit, diagnostics
