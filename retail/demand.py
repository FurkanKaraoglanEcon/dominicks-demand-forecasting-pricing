"""Unrestricted observational responses, directly linked prices, and purchasing dynamics.

Inputs: prepared cola panel and configured specifications.
Outputs: fitted demand systems, joint covariance, elasticities and matched dynamic samples.
"""

from dataclasses import dataclass
import logging
from itertools import product
import numpy as np
import pandas as pd
from .io import SEASON, PRICES, PROMOTIONS, save_csv, save_json, calendar_lag
from .econometrics import fit_system

LOG = logging.getLogger(__name__)


@dataclass
class DemandModel:
    system: str
    specification: str
    fit: object
    elasticity: np.ndarray
    price_covariance: np.ndarray
    baseline_units: np.ndarray
    price_indices: list
    dynamic: bool = False


def system_definition(system):
    """Return named outcomes and price indices for the declared aggregation."""
    if system == "family":
        return ["flq1", "flq2"], ["flp1", "flp2"]
    if system == "linked_product":
        return [f"lq{j}" for j in range(1, 5)], ["flp1", "flp2"]
    return [f"lq{j}" for j in range(1, 5)], PRICES


def make_model(
    panel,
    system,
    specification,
    regressors,
    price_indices,
    sample,
    week_effects,
    dynamic=False,
):
    """Fit the declared panel sample and return slopes, joint covariance and fitted mean quantities. Price-index positions specify the current/cumulative contrast."""
    outcomes, _ = system_definition(system)
    fit = fit_system(panel, outcomes, regressors, sample, week_effects)
    response_count, price_count = len(outcomes), len(price_indices[0])
    # Rows of this contrast map all distributed-lag coefficients to a persistent-price response.
    contrast = np.zeros(
        (response_count * price_count, len(regressors) * response_count)
    )
    # This numerical contrast sums current/lag slopes in each equation block.
    for equation, price in product(range(response_count), range(price_count)):
        indices = [
            equation * len(regressors) + lag_indices[price]
            for lag_indices in price_indices
        ]
        contrast[equation * price_count + price, indices] = 1
    elasticity = (contrast @ fit.beta.T.reshape(-1)).reshape(
        response_count, price_count
    )
    covariance = contrast @ fit.covariance @ contrast.T
    baseline = np.full((len(panel), response_count), np.nan)
    baseline[fit.mask] = np.exp(fit.fitted) * np.exp(fit.residual).mean(axis=0)
    return DemandModel(
        system,
        specification,
        fit,
        elasticity,
        covariance,
        baseline,
        price_indices,
        dynamic,
    )


def coefficient_rows(model):
    """Convert a DemandModel to long elasticity and two-way confidence-interval records."""
    rows = []
    for equation in range(model.elasticity.shape[0]):
        for price in range(model.elasticity.shape[1]):
            index = equation * model.elasticity.shape[1] + price
            se = np.sqrt(max(0, model.price_covariance[index, index]))
            b = model.elasticity[equation, price]
            rows.append(
                dict(
                    system=model.system,
                    specification=model.specification,
                    demand=equation + 1,
                    price=price + 1,
                    elasticity=b,
                    se=se,
                    lower95=b - model.fit.critical * se,
                    upper95=b + model.fit.critical * se,
                    N=model.fit.n,
                    stores=model.fit.stores,
                    weeks=model.fit.weeks,
                    inference="two_way_store_week",
                    response="cumulative" if model.dynamic else "static",
                )
            )
    return rows


def dynamic_rows(model, static):
    """Return immediate, subsequent and cumulative lag contrasts and their joint delta inference."""
    fit = model.fit
    response_count, price_count = model.elasticity.shape
    rows = []
    responses = [
        ("immediate", [0]),
        ("subsequent", list(range(1, len(model.price_indices)))),
        ("cumulative", list(range(len(model.price_indices)))),
    ]
    for equation, price, (label, positions) in product(
        range(response_count), range(price_count), responses
    ):
        contrast = np.zeros(len(fit.regressors) * response_count)
        for lag in positions:
            contrast[
                equation * len(fit.regressors) + model.price_indices[lag][price]
            ] = 1
        b = contrast @ fit.beta.T.reshape(-1)
        se = np.sqrt(max(0, contrast @ fit.covariance @ contrast))
        rows.append(
            dict(
                system=model.system,
                demand=equation + 1,
                price=price + 1,
                response=label,
                elasticity=b,
                se=se,
                lower95=b - fit.critical * se,
                upper95=b + fit.critical * se,
                aligned_static=static.elasticity[equation, price],
                N=fit.n,
                stores=fit.stores,
            )
        )
    return rows


def estimate_income_heterogeneity(panel, output):
    """Fit own-price/income interactions on finite census rows and save the matched-sample effects."""
    rows = []
    for j in range(1, 5):
        frame = panel.copy()
        frame["own_price_income"] = frame[f"lp{j}"] * frame.income_c
        fit = fit_system(
            frame,
            [f"lq{j}"],
            PRICES + PROMOTIONS + ["own_price_income"],
            np.ones(len(panel), bool),
            True,
        )
        index = len(fit.regressors) - 1
        se = np.sqrt(max(0, fit.covariance[index, index]))
        rows.append(
            dict(
                product=j,
                own_elasticity=fit.beta[j - 1, 0],
                income_interaction=fit.beta[index, 0],
                se_interaction=se,
                interaction=fit.beta[index, 0],
                se=se,
                lower95=fit.beta[index, 0] - fit.critical * se,
                upper95=fit.beta[index, 0] + fit.critical * se,
                N=fit.n,
                stores=fit.stores,
            )
        )
    save_csv(rows, output, "income_heterogeneity.csv")


def common_relative_effects(models, weights, output):
    """Decompose unrestricted slopes into a common brand move and a regular/diet wedge."""
    rows = []
    for key, model in models.items():
        if model.system != "product":
            continue
        beta = model.elasticity.reshape(-1)
        for equation, (family, (a, b)) in product(
            range(4), enumerate([(0, 1), (2, 3)], 1)
        ):
            common = np.zeros(16)
            common[[equation * 4 + a, equation * 4 + b]] = 1
            relative = np.zeros(16)
            relative[equation * 4 + a] = weights[b]
            relative[equation * 4 + b] = -weights[a]
            common_se = np.sqrt(max(0, common @ model.price_covariance @ common))
            relative_se = np.sqrt(max(0, relative @ model.price_covariance @ relative))
            rows.append(
                dict(
                    specification=key[1],
                    product=equation + 1,
                    family=family,
                    common_price_elasticity=common @ beta,
                    relative_price_slope=relative @ beta,
                    common_se=common_se,
                    relative_se=relative_se,
                    common_lower95=common @ beta - model.fit.critical * common_se,
                    common_upper95=common @ beta + model.fit.critical * common_se,
                    relative_lower95=relative @ beta - model.fit.critical * relative_se,
                    relative_upper95=relative @ beta + model.fit.critical * relative_se,
                    N=model.fit.n,
                    response="cumulative" if model.dynamic else "static",
                )
            )
    save_csv(rows, output, "common_and_relative_price_effects.csv")


def estimate(panel, config):
    """Fit all declared primary systems from panel/config and return models keyed by system/specification; export samples, inference and diagnostics."""
    output = config["paths"]["output"] / "demand"
    output.mkdir(parents=True, exist_ok=True)
    lag_count = config["demand"]["dynamic_lags"]
    models, coefficients, dynamics, fit_rows, diagnostics = {}, [], [], [], []
    all_rows = np.ones(len(panel), bool)
    for system in ["product", "linked_product", "family"]:
        outcomes, prices = system_definition(system)
        for specification in config["demand"]["specifications"]:
            calendar = (
                []
                if specification == "week_FE"
                else SEASON + (["trend"] if specification == "season_trend" else [])
            )
            regressors = prices + PROMOTIONS + calendar
            model = make_model(
                panel,
                system,
                specification,
                regressors,
                [list(range(len(prices)))],
                all_rows,
                specification == "week_FE",
            )
            models[(system, specification)] = model
            coefficients.extend(coefficient_rows(model))
        if system != "family":
            regressors = (
                prices + PROMOTIONS + ["omitted_log_price", "omitted_weight_coverage"]
            )
            model = make_model(
                panel,
                system,
                "omitted_index",
                regressors,
                [list(range(len(prices)))],
                all_rows,
                True,
            )
            models[(system, "omitted_index")] = model
            coefficients.extend(coefficient_rows(model))
            price_terms = [prices] + [
                [f"{p}_lag{lag}" for p in prices] for lag in range(1, lag_count + 1)
            ]
            promo_terms = PROMOTIONS + [
                f"{p}_lag{lag}" for lag in range(1, lag_count + 1) for p in PROMOTIONS
            ]
            regressors = [p for group in price_terms for p in group] + promo_terms
            positions = [
                list(range(lag * len(prices), (lag + 1) * len(prices)))
                for lag in range(lag_count + 1)
            ]
            model = make_model(
                panel,
                system,
                "distributed_lag",
                regressors,
                positions,
                all_rows,
                True,
                True,
            )
            static = make_model(
                panel,
                system,
                "aligned_static",
                prices + PROMOTIONS,
                [list(range(len(prices)))],
                model.fit.mask,
                True,
            )
            models[(system, "distributed_lag")] = model
            coefficients.extend(coefficient_rows(model))
            coefficients.extend(coefficient_rows(static))
            dynamics.extend(dynamic_rows(model, static))
            # A lead association is a diagnostic for anticipation/confounding, not an instrument.
            lead_frame = panel.copy()
            lead_names = []
            for p in prices:
                name = p + "_lead1"
                lead_frame[name] = calendar_lag(panel, p, -1)
                lead_names.append(name)
            lead = fit_system(
                lead_frame, outcomes, prices + lead_names + PROMOTIONS, all_rows, True
            )
            diagnostics.extend(
                lead_coefficient_rows(lead, system, len(prices), lead_names)
            )
    for key, model in models.items():
        fit = model.fit
        for j in range(len(fit.outcomes)):
            y = panel.loc[fit.mask, fit.outcomes[j]].to_numpy()
            fit_rows.append(
                dict(
                    system=key[0],
                    specification=key[1],
                    product=j + 1,
                    N=fit.n,
                    stores=fit.stores,
                    r2=1
                    - np.sum(fit.residual[:, j] ** 2) / np.sum((y - y.mean()) ** 2),
                    rmse=fit.rmse[j],
                    smear=np.exp(fit.residual[:, j]).mean(),
                    parameters=fit.parameters,
                )
            )
        # A portable numerical model artifact supports independent inspection of joint uncertainty.
        np.savez_compressed(
            output / f"model_{key[0]}_{key[1]}.npz",
            elasticity=model.elasticity,
            covariance=model.price_covariance,
            beta=fit.beta,
            regressors=np.array(fit.regressors),
            mask=fit.mask,
        )
    table = save_csv(coefficients, output, "demand_sensitivity.csv")
    save_csv(dynamics, output, "demand_dynamics.csv")
    save_csv(diagnostics, output, "price_lead_diagnostics.csv")
    save_csv(fit_rows, output, "demand_fit.csv")
    legacy = table.loc[
        table.system.eq("product")
        & table.specification.isin(["week_FE", "season_trend"])
    ].rename(
        columns={
            "demand": "demand_product",
            "price": "price_product",
            "N": "observations",
        }
    )
    save_csv(
        legacy[
            [
                "demand_product",
                "price_product",
                "specification",
                "elasticity",
                "se",
                "lower95",
                "upper95",
                "observations",
            ]
        ],
        output,
        "elasticities.csv",
    )
    estimate_income_heterogeneity(panel, output)
    weights = (
        pd.read_csv(config["paths"]["output"] / "family_weights.csv")
        .sort_values("product")
        .weight.to_numpy()
    )
    common_relative_effects(models, weights, output)
    save_json(
        dict(
            inference="Cameron-Gelbach-Miller store + week - store-week covariance; t(min(clusters)-1)",
            signs_restricted=False,
            instruments_used=[],
            estimation_scope="observed positive sales; all available weeks",
            dynamics="four price/promotion lags; cumulative response to a persistent change; no lagged dependent variable",
        ),
        output / "estimation_notes.json",
    )
    LOG.info("Estimated %s demand systems; lag-aligned samples documented", len(models))
    return models


def lead_coefficient_rows(lead, system, price_count, lead_names):
    """Return future-price associations as anticipation/confounding diagnostics."""
    rows = []
    for equation, (k, name) in product(
        range(len(lead.outcomes)), enumerate(lead_names)
    ):
        index = equation * len(lead.regressors) + price_count + k
        rows.append(
            dict(
                system=system,
                demand=equation + 1,
                price=k + 1,
                lead_elasticity=lead.beta[price_count + k, equation],
                se=np.sqrt(max(0, lead.covariance[index, index])),
                N=lead.n,
            )
        )
    return rows
