"""Estimate product, family, dynamic, income, visit and category demand.

Inputs: the common four-price panels and visible product mappings.
Outputs: coefficients and two-way cluster inference, sample/identification
diagnostics, mean-scale elasticities, and serialized main models for pricing.
All estimates are observational; no experimental assignment is supplied.
"""

import pickle
import numpy as np
import pandas as pd

from .io import stage_log, table_path
from .equation_inference import (
    absorb,
    coefficient_rows,
    fit_iv,
    fit_ols,
)

from .product_definitions import PRODUCTS, PRICE_NAMES, PROMOTION_NAMES


def demand_regressors(panel, price_form="log"):
    """Return the explicitly ordered four prices and four recorded promotions."""
    prices = panel[PRICE_NAMES].to_numpy()
    if price_form == "log":
        prices = np.log(prices)
    return (
        np.column_stack([prices, panel[PROMOTION_NAMES].to_numpy()]),
        PRICE_NAMES + PROMOTION_NAMES,
    )


def estimate_products(panel, category, model, transformation, extra=None):
    """Estimate separate product equations on a shared, explicit sample.

    Return fitted equations and a long coefficient table. Positive sales are
    required for log outcomes; log1p retains genuine priced zero-sales rows.
    """
    sample = panel.copy()
    if transformation != "log1p":
        sample = sample[
            sample[[f"units_{p}" for p in PRODUCTS]].gt(0).all(axis=1)
        ].copy()
    x, names = demand_regressors(sample, "level" if model == "semilog" else "log")
    if extra == "income":
        sample = sample[sample.income_centered.notna()].copy()
        x, names = demand_regressors(sample)
        interaction = (
            np.log(sample[PRICE_NAMES].to_numpy())
            * sample.income_centered.to_numpy()[:, None]
        )
        x = np.column_stack([x, interaction])
        names = names + [f"income_price_{p}" for p in PRODUCTS]
    if extra == "traffic_control":
        x = np.column_stack([x, np.log(sample.customers)])
        names = names + ["log_customers"]
    fits = {}
    rows = []
    for product in PRODUCTS:
        y = (
            np.log1p(sample[f"units_{product}"])
            if transformation == "log1p"
            else np.log(sample[f"units_{product}"])
        )
        if transformation == "per_visit":
            y = y - np.log(sample.customers)
        fit = fit_ols(y, x, names, sample.store, sample.week)
        fit["sample_index"] = sample.index.to_numpy()
        fit["transformation"] = transformation
        fit["price_form"] = "level" if model == "semilog" else "log"
        # A store-specific Duan factor accommodates persistent variance differences.
        smearing = (
            pd.Series(np.exp(fit["residuals"]), index=sample.index)
            .groupby(sample.store)
            .transform("mean")
        )
        fit["smearing"] = smearing.to_numpy()
        fits[product] = fit
        rows.extend(coefficient_rows(fit, category, model, product))
    return fits, rows


def dynamic_equations(panel, category):
    """Estimate one-week price effects and quantity persistence on consecutive weeks."""
    data = panel.sort_values(["store", "week"]).copy()
    consecutive = data.groupby("store").week.diff().eq(1)
    lag_prices = (
        np.log(data[PRICE_NAMES])
        .groupby(data.store)
        .shift(1)
        .where(consecutive, axis=0)
    )
    lag_quantities = (
        np.log(
            data[[f"units_{p}" for p in PRODUCTS]].where(
                data[[f"units_{p}" for p in PRODUCTS]].gt(0)
            )
        )
        .groupby(data.store)
        .shift(1)
        .where(consecutive, axis=0)
    )
    data[[f"lag_price_{p}" for p in PRODUCTS]] = lag_prices
    data[[f"lag_quantity_{p}" for p in PRODUCTS]] = lag_quantities
    valid = data[[f"units_{p}" for p in PRODUCTS]].gt(0).all(axis=1)
    valid &= (
        data[
            [f"lag_price_{p}" for p in PRODUCTS]
            + [f"lag_quantity_{p}" for p in PRODUCTS]
        ]
        .notna()
        .all(axis=1)
    )
    data = data[valid].copy()
    _, rows = estimate_products(data, category, "static_dynamic_sample", "log")
    long_run = []
    for product in PRODUCTS:
        x, names = demand_regressors(data)
        names = names + [f"lag_price_{p}" for p in PRODUCTS] + ["lag_quantity_own"]
        x = np.column_stack(
            [
                x,
                data[[f"lag_price_{p}" for p in PRODUCTS]],
                data[f"lag_quantity_{product}"],
            ]
        )
        fit = fit_ols(np.log(data[f"units_{product}"]), x, names, data.store, data.week)
        rows.extend(coefficient_rows(fit, category, "dynamic", product))
        values = dict(zip(fit["names"], fit["coefficients"]))
        rho = values["lag_quantity_own"]
        for price in PRODUCTS:
            gradient = np.zeros(len(fit["names"]))
            current_index = fit["names"].index(f"price_{price}")
            lag_index = fit["names"].index(f"lag_price_{price}")
            rho_index = fit["names"].index("lag_quantity_own")
            total = values[f"price_{price}"] + values[f"lag_price_{price}"]
            if abs(rho) < 1:
                gradient[current_index] = 1 / (1 - rho)
                gradient[lag_index] = 1 / (1 - rho)
                gradient[rho_index] = total / (1 - rho) ** 2
                variance = gradient @ fit["covariance"] @ gradient
                elasticity = total / (1 - rho)
            else:
                elasticity, variance = np.nan, np.nan
            long_run.append(
                dict(
                    category=category,
                    outcome=product,
                    price_product=price,
                    persistence=rho,
                    stable=abs(rho) < 1,
                    long_run_elasticity=elasticity,
                    delta_standard_error=(
                        np.sqrt(max(variance, 0)) if np.isfinite(variance) else np.nan
                    ),
                )
            )
    return rows, long_run


def family_equations(panel, config):
    """Estimate cola brand and diet families with fixed training quantity weights.

    All members are two-liter packages. The geometric price index uses fixed
    training shares; current sales never determine the price index weights.
    """
    groups = {
        "brand": {"Pepsi": ["p1", "p2"], "Coke": ["p3", "p4"]},
        "type": {"Regular": ["p1", "p3"], "Diet": ["p2", "p4"]},
    }
    train = panel[panel.week.le(config["sample"]["train_end"])]
    rows = []
    weights_table = []
    for partition, families in groups.items():
        data = panel[["store", "week"]].copy()
        for family, products in families.items():
            weights = train[[f"units_{p}" for p in products]].sum().to_numpy()
            weights = weights / weights.sum()
            data[f"quantity_{family}"] = panel[[f"units_{p}" for p in products]].sum(
                axis=1
            )
            data[f"price_{family}"] = (
                np.log(panel[[f"price_{p}" for p in products]]).to_numpy() @ weights
            )
            data[f"promotion_{family}"] = (
                panel[[f"promotion_{p}" for p in products]].to_numpy() @ weights
            )
            weights_table.extend(
                dict(partition=partition, family=family, product=p, weight=w)
                for p, w in zip(products, weights)
            )
        data = data[data[[f"quantity_{f}" for f in families]].gt(0).all(axis=1)]
        names = [f"price_{f}" for f in families] + [f"promotion_{f}" for f in families]
        x = data[names].to_numpy()
        rows.extend(estimate_family_partition(data, families, names, x, partition))
    return rows, weights_table


def estimate_family_partition(data, families, names, x, partition):
    """Fit both family equations for one named partition of the four colas."""
    rows = []
    for family in families:
        fit = fit_ols(
            np.log(data[f"quantity_{family}"]), x, names, data.store, data.week
        )
        rows.extend(coefficient_rows(fit, "sdr", f"family_{partition}", family))
    return rows


def identification_audit(panel, category):
    """Audit residual price support, AAC IV relevance and other-store instruments."""
    data = panel[panel[[f"units_{p}" for p in PRODUCTS]].gt(0).all(axis=1)].copy()
    log_prices = np.log(data[PRICE_NAMES])
    within, _, _ = absorb(log_prices, [data.store, data.week])
    diagnostics = []
    price_correlations = np.corrcoef(within.T)
    # Under a common week FE, a leave-one-store-out mean can be an almost
    # mechanical negative multiple of the endogenous price. Do not use it as IV.
    week_sum = log_prices.groupby(data.week).transform("sum")
    week_count = data.groupby("week").week.transform("count")
    other_valid = week_count.gt(1)
    other_store = (week_sum[other_valid] - log_prices[other_valid]).div(
        week_count[other_valid] - 1, axis=0
    )
    other_within, _, _ = absorb(
        other_store, [data.loc[other_valid, "store"], data.loc[other_valid, "week"]]
    )
    comparable_within, _, _ = absorb(
        log_prices[other_valid],
        [data.loc[other_valid, "store"], data.loc[other_valid, "week"]],
    )
    for index, product in enumerate(PRODUCTS):
        diagnostics.append(
            dict(
                category=category,
                product=product,
                log_price_sd=log_prices.iloc[:, index].std(),
                residual_log_price_sd=within[:, index].std(),
                variance_remaining=np.var(within[:, index])
                / np.var(log_prices.iloc[:, index]),
                largest_residual_cross_price_correlation=max(
                    abs(np.delete(price_correlations[index], index))
                ),
                other_store_diagnostic_rows=int(other_valid.sum()),
                single_store_rows_excluded=int((~other_valid).sum()),
                other_store_within_correlation=np.corrcoef(
                    comparable_within[:, index], other_within[:, index]
                )[0, 1],
            )
        )
    cost_sample = data[data[[f"cost_{p}" for p in PRODUCTS]].gt(0).all(axis=1)].copy()
    _, rows = estimate_products(cost_sample, category, "OLS_IV_sample", "log")
    x, names = demand_regressors(cost_sample)
    costs = np.log(cost_sample[[f"cost_{p}" for p in PRODUCTS]].to_numpy())
    stages = []
    for product in PRODUCTS:
        fit, first_stages = fit_iv(
            np.log(cost_sample[f"units_{product}"]),
            x[:, :4],
            x[:, 4:],
            costs,
            names,
            cost_sample.store.to_numpy(),
            cost_sample.week.to_numpy(),
        )
        rows.extend(coefficient_rows(fit, category, "AAC_IV_sensitivity", product))
        if product == "p1":
            stages.extend(
                dict(category=category, n=fit["n"], **stage) for stage in first_stages
            )
    return rows, diagnostics, stages


def category_equations(panel, category):
    """Measure portfolio and full-file package volume responses, conditional on mix."""
    rows = []
    outcomes = ["portfolio_units", "category_units"]
    if category == "sdr":
        outcomes.append("two_liter_units")
    for outcome in outcomes:
        data = panel[panel[outcome].gt(0) & panel.active_upcs.gt(0)]
        x, names = demand_regressors(data)
        x = np.column_stack([x, np.log(data.active_upcs)])
        names = names + ["log_active_upcs"]
        fit = fit_ols(np.log(data[outcome]), x, names, data.store, data.week)
        rows.extend(coefficient_rows(fit, category, "category_expansion", outcome))
    return rows


def mean_scale_elasticities(fits, panel, category, model):
    """Convert transformed slopes to elasticities of predicted mean package sales."""
    rows = []
    for product, fit in fits.items():
        mean_quantity = np.exp(fit["fitted"]) * fit["smearing"]
        if fit["transformation"] == "log1p":
            mean_quantity = mean_quantity - 1
        assert (mean_quantity > 0).all()
        sample = panel.loc[fit["sample_index"]]
        for price in PRODUCTS:
            index = fit["names"].index(f"price_{price}")
            factor = np.ones(len(sample))
            if fit["price_form"] == "level":
                factor = sample[f"price_{price}"].to_numpy()
            if fit["transformation"] == "log1p":
                factor *= (mean_quantity + 1) / mean_quantity
            average_factor = factor.mean()
            rows.append(
                dict(
                    category=category,
                    model=model,
                    outcome=product,
                    price_product=price,
                    elasticity=fit["coefficients"][index] * average_factor,
                    delta_standard_error=np.sqrt(fit["covariance"][index, index])
                    * average_factor,
                )
            )
    return rows


def income_quartile_effects(fits, panel, category):
    """Return own-price effects and delta SEs at three store-income quantiles."""
    income_values = (
        panel[["store", "income_centered"]]
        .drop_duplicates()
        .income_centered.quantile([0.25, 0.5, 0.75])
    )
    rows = []
    for product, fit in fits.items():
        own = fit["names"].index(f"price_{product}")
        interaction = fit["names"].index(f"income_price_{product}")
        for quantile, income in income_values.items():
            gradient = np.zeros(len(fit["names"]))
            gradient[own], gradient[interaction] = 1, income
            rows.append(
                dict(
                    category=category,
                    outcome=product,
                    income_quantile=quantile,
                    centered_log_income=income,
                    own_elasticity=gradient @ fit["coefficients"],
                    standard_error=np.sqrt(
                        max(gradient @ fit["covariance"] @ gradient, 0)
                    ),
                )
            )
    return rows


def run_auxiliary_demand(config):
    """Execute every demand exercise and save main models for the pricing stage."""
    logger = stage_log(config, "demand")
    rows, elasticities, diagnostics, stages, dynamic, samples, income_effects = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for category in config["categories"]:
        panel = pd.read_parquet(
            config["paths"]["processed"] / f"panel_{category}.parquet"
        )
        main = {}
        for model, transformation in [
            ("loglog", "log"),
            ("log1p", "log1p"),
            ("semilog", "log"),
        ]:
            fits, coefficients = estimate_products(
                panel, category, model, transformation
            )
            main[model] = fits
            rows.extend(coefficients)
            elasticities.extend(mean_scale_elasticities(fits, panel, category, model))
            logger.info(
                "%s %s: %s common markets; condition number %.2f",
                category,
                model,
                fits["p1"]["n"],
                fits["p1"]["condition_number"],
            )
        positive = panel[panel[[f"units_{p}" for p in PRODUCTS]].gt(0).all(axis=1)]
        _, coefficients = estimate_products(
            positive, category, "log1p_common_positive", "log1p"
        )
        rows.extend(coefficients)
        income_fits, coefficients = estimate_products(
            panel, category, "income_interactions", "log", extra="income"
        )
        rows.extend(coefficients)
        _, coefficients = estimate_products(
            positive[positive.income.notna()], category, "OLS_income_sample", "log"
        )
        rows.extend(coefficients)
        income_effects.extend(income_quartile_effects(income_fits, panel, category))
        visit_sample = positive[positive.customers.notna()]
        for model, transform, extra in [
            ("OLS_visit_sample", "log", None),
            ("per_visit", "per_visit", None),
            ("traffic_control", "log", "traffic_control"),
        ]:
            _, coefficients = estimate_products(
                visit_sample, category, model, transform, extra=extra
            )
            rows.extend(coefficients)
        coefficients, long_run = dynamic_equations(panel, category)
        rows.extend(coefficients)
        dynamic.extend(long_run)
        coefficients, price_diagnostics, first_stages = identification_audit(
            panel, category
        )
        rows.extend(coefficients)
        diagnostics.extend(price_diagnostics)
        stages.extend(first_stages)
        rows.extend(category_equations(panel, category))
        with open(
            config["paths"]["output"] / "models" / f"demand_{category}.pkl", "wb"
        ) as stream:
            pickle.dump(main, stream)
        for model, fits in main.items():
            samples.extend(
                dict(
                    category=category,
                    model=model,
                    outcome=p,
                    n=f["n"],
                    stores=f["stores"],
                    weeks=f["weeks"],
                    first_week=min(f["week"]),
                    last_week=max(f["week"]),
                    absorption_iterations=f["iterations"],
                    absorption_error=f["absorption_error"],
                    dropped_terms=";".join(f["dropped"]),
                )
                for p, f in fits.items()
            )
        if category == "sdr":
            family_rows, weights = family_equations(panel, config)
            rows.extend(family_rows)
            pd.DataFrame(weights).to_csv(
                table_path(config, "family_price_weights"), index=False
            )
        logger.info(
            "%s finished: functional forms, income, visits, dynamics, IV audit and category volume",
            category,
        )
    pd.DataFrame(rows).to_csv(table_path(config, "demand_coefficients"), index=False)
    pd.DataFrame(elasticities).to_csv(table_path(config, "elasticities"), index=False)
    pd.DataFrame(diagnostics).to_csv(
        table_path(config, "price_identification"), index=False
    )
    pd.DataFrame(stages).to_csv(table_path(config, "IV_first_stages"), index=False)
    pd.DataFrame(dynamic).to_csv(table_path(config, "dynamic_long_run"), index=False)
    pd.DataFrame(samples).to_csv(table_path(config, "demand_samples"), index=False)
    pd.DataFrame(income_effects).to_csv(
        table_path(config, "income_elasticities"), index=False
    )
