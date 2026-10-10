"""Transparent within OLS, joint equation covariance and level retransformation.

Inputs: named panel outcomes/regressors and explicit sample masks.
Outputs: within-OLS fits, joint clustered inference and past-only mean-scale forecasts.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def fixed_effect_rank(store, week):
    """Rank of store/week dummies, allowing disconnected panel components."""
    stores = np.unique(store, return_inverse=True)[1]
    weeks = np.unique(week, return_inverse=True)[1]
    ns, nw = stores.max() + 1, weeks.max() + 1
    graph = sparse.coo_matrix(
        (np.ones(len(stores)), (stores, ns + weeks)), shape=(ns + nw, ns + nw)
    )
    components = connected_components(
        graph + graph.T, directed=False, return_labels=False
    )
    return int(ns + nw - components)


def demean(values, groups):
    """Return values minus column means within groups, preserving row order."""
    matrix = np.asarray(values, dtype=float)
    return (
        matrix
        - pd.DataFrame(matrix).groupby(np.asarray(groups)).transform("mean").to_numpy()
    )


def absorb(values, stores, weeks=None, tolerance=1e-11):
    """Return values residualized on stores and optional weeks.
    Alternate projections to tolerance; raise if they do not converge."""
    residual = np.asarray(values, dtype=float).copy()
    if weeks is None:
        return demean(residual, stores)
    for _ in range(500):
        previous = residual.copy()
        residual = demean(demean(residual, stores), weeks)
        if np.max(np.abs(residual - previous)) < tolerance:
            return residual
    raise RuntimeError("Store/week projections did not converge")


@dataclass
class Fit:
    outcomes: list
    regressors: list
    mask: np.ndarray
    beta: np.ndarray
    residual: np.ndarray
    fitted: np.ndarray
    within_x: np.ndarray
    bread: np.ndarray
    covariance: np.ndarray
    store_covariance: np.ndarray
    n: int
    stores: int
    weeks: int
    parameters: int
    critical: float
    rmse: np.ndarray


def score_covariance(scores, bread, groups, n, parameters):
    """Aggregate scores by groups and apply bread on both sides.
    Return clustered covariance with corrections based on n and parameters."""
    sums = pd.DataFrame(scores).groupby(np.asarray(groups)).sum().to_numpy()
    clusters = len(sums)
    if clusters < 2:
        raise ValueError("At least two clusters are required")
    correction = clusters / (clusters - 1) * (n - 1) / (n - parameters)
    return bread @ (sums.T @ sums) @ bread * correction


def fit_system(panel, outcomes, regressors, sample, week_effects=False, inference=True):
    """Fit named outcomes on regressors for panel rows admitted by sample.
    Always absorb store effects; week_effects adds week effects. With inference,
    return a Fit containing joint store/week covariance, masks, fitted values,
    residuals, slopes and sample/rank information; otherwise omit covariance work."""
    mask = np.asarray(sample, dtype=bool) & np.isfinite(
        panel[outcomes + regressors + ["store", "week"]].to_numpy(float)
    ).all(axis=1)
    data = panel.loc[mask]
    x, y = data[regressors].to_numpy(float), data[outcomes].to_numpy(float)
    z = absorb(np.column_stack([x, y]), data.store, data.week if week_effects else None)
    wx, wy = z[:, : len(regressors)], z[:, len(regressors) :]
    beta, _, rank, _ = np.linalg.lstsq(wx, wy, rcond=None)
    if rank != len(regressors):
        raise RuntimeError(f"Rank deficient: {outcomes}, {regressors}")
    residual = wy - wx @ beta
    n, stores, weeks = len(data), data.store.nunique(), data.week.nunique()
    parameters = len(regressors) + (
        fixed_effect_rank(data.store, data.week) if week_effects else stores
    )
    if n <= parameters:
        raise ValueError("Insufficient residual degrees of freedom")
    bread = np.linalg.inv(wx.T @ wx)
    dimension = len(regressors) * len(outcomes)
    covariance = store_covariance = np.empty((0, 0))
    if inference:
        # Equation blocks preserve contemporaneous cross-product error covariance.
        scores = np.column_stack(
            [wx * residual[:, j, None] for j in range(len(outcomes))]
        )
        joint_bread = np.kron(np.eye(len(outcomes)), bread)
        store_covariance = score_covariance(
            scores, joint_bread, data.store, n, parameters
        )
        week_covariance = score_covariance(
            scores, joint_bread, data.week, n, parameters
        )
        # Store-week intersections are unique rows; the third term is HC1.
        intersection = (
            joint_bread @ (scores.T @ scores) @ joint_bread * n / (n - parameters)
        )
        covariance = store_covariance + week_covariance - intersection
        covariance = (covariance + covariance.T) / 2
        assert covariance.shape == (dimension, dimension)
        assert (
            np.diag(covariance).min() >= -1e-10
        ), "Negative two-way variance; inspect inference"
    return Fit(
        outcomes,
        regressors,
        mask,
        beta,
        residual,
        y - residual,
        wx,
        bread,
        covariance,
        store_covariance,
        n,
        stores,
        weeks,
        parameters,
        t.ppf(0.975, min(stores, weeks) - 1),
        np.sqrt((residual**2).sum(axis=0) / (n - parameters)),
    )


def forecast_levels(panel, outcome, regressors, fitting_sample, prediction_sample):
    """Fit outcome on regressors using fitting_sample and predict prediction_sample.
    Return log1p mean-scale forecasts, the Fit and training-residual smearing factor."""
    fit = fit_system(panel, [outcome], regressors, fitting_sample, inference=False)
    training = panel.loc[fit.mask]
    effects = (
        pd.Series(
            training[outcome].to_numpy()
            - training[regressors].to_numpy() @ fit.beta[:, 0],
            index=training.index,
        )
        .groupby(training.store)
        .mean()
    )
    target = panel.loc[prediction_sample]
    linear = (
        target[regressors].to_numpy() @ fit.beta[:, 0]
        + target.store.map(effects).to_numpy()
    )
    smear = np.exp(fit.residual[:, 0]).mean()
    # E[exp(log(1+q))] - 1; subtracting one before smearing changes the estimand.
    prediction = np.maximum(0, np.exp(linear) * smear - 1)
    return prediction, fit, smear


def metrics(actual, predicted):
    """Return package-sales MAE, RMSE, WAPE and percentage bias for actual/predicted."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error, total = predicted - actual, actual.sum()
    return dict(
        N=len(actual),
        MAE=np.abs(error).mean(),
        RMSE=np.sqrt((error**2).mean()),
        WAPE=100 * np.abs(error).sum() / total if total > 0 else np.nan,
        bias_pct=100 * error.sum() / total if total > 0 else np.nan,
    )


def group_sum(values, codes):
    """Sum a matrix by integer group code for cluster-score diagnostics."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return np.column_stack(
        [np.bincount(codes, weights=values[:, j]) for j in range(values.shape[1])]
    )
