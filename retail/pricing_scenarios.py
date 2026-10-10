"""Execute store-specific pricing cases and aggregate the same policy records.

Inputs: fitted demand systems, reference rows and common uncertainty draws.
Outputs: store/product policies, aggregate profits, feasibility and search audits.
Prediction, objective evaluation and search are imported from pricing.py.
"""

from itertools import product
import logging
import numpy as np
import pandas as pd
from .io import save_csv, save_json
from .pricing import (
    PAIRS,
    predict_units,
    profit,
    feasible,
    support_for_store,
    linked_grid,
    optimize,
    model_inputs,
    common_reference,
    profit_gradient,
)
from .pricing_uncertainty import parameter_draws, drawn_gains, robust_screen
from .profit_diagnostics import full_panel_derivatives

LOG = logging.getLogger(__name__)


def run_pricing(panel, models, config):
    """Use panel reference observations and models under config policies/costs.
    Write supported store/aggregate searches and uncertainty; return no data."""
    output = config["paths"]["output"] / "pricing"
    output.mkdir(parents=True, exist_ok=True)
    settings = config["pricing"]
    weights = (
        pd.read_csv(config["paths"]["output"] / "family_weights.csv")
        .sort_values("product")
        .weight.to_numpy()
    )
    week, references = common_reference(
        panel, models, settings["minimum_reference_stores"]
    )
    history = panel.loc[panel.week.le(week)]
    save_csv(
        references[["store", "week"]].assign(reference_stores=len(references)),
        output,
        "pricing_reference_stores.csv",
    )
    rng = np.random.default_rng(settings["seed"])
    draws, uncertainty_audit = {}, []
    for key, model in models.items():
        draws[key], audit = parameter_draws(model, settings, rng, weights)
        uncertainty_audit.append(audit)
    save_csv(uncertainty_audit, output, "uncertainty_covariance_audit.csv")
    summaries, details, uncertainty, support_records, derivatives = [], [], [], [], []
    aggregate_draws = {}
    cases = []
    for key, model in models.items():
        rules = (
            ["product_free", "pair_linked"]
            if model.system == "product"
            else ["pair_linked"]
        )
        cases.extend((key, model, rule) for rule in rules)
    for key, model, rule in cases:
        LOG.info("Pricing %s / %s / %s (%s stores)", *key, rule, len(references))
        store_rows, product_rows, uncertainty_rows, support_rows, derivative_rows = (
            price_one_model(
                key,
                model,
                rule,
                references,
                history,
                weights,
                settings,
                draws,
                aggregate_draws,
            )
        )
        summaries.extend(store_rows)
        details.extend(product_rows)
        uncertainty.extend(uncertainty_rows)
        support_records.extend(support_rows)
        derivatives.extend(derivative_rows)
    # Every policy must export all four product contributions. Publishing a
    # completed temporary file avoids exposing a partially written table.
    assert len(details) == 4 * len(summaries)
    store_results = save_csv(summaries, output, "pricing_sensitivity_store.csv")
    product_results = save_csv(details, output, "pricing_sensitivity_details.csv")
    exported_details = pd.read_csv(output / "pricing_sensitivity_details.csv")
    assert len(exported_details) == len(
        product_results
    ), "Incomplete product-policy export"
    keys = [
        "system",
        "specification",
        "price_rule",
        "maxchange",
        "cost_multiplier",
        "policy",
    ]
    aggregate = store_results.groupby(keys, as_index=False).agg(
        profit=("profit", "sum"),
        baseline_profit=("baseline_profit", "sum"),
        stores=("store", "nunique"),
        converged=("converged", "mean"),
    )
    aggregate["gain_pct"] = np.where(
        aggregate.baseline_profit.gt(0),
        100 * (aggregate.profit / aggregate.baseline_profit - 1),
        np.nan,
    )
    boundaries = product_results.groupby(keys, as_index=False).agg(
        bound_share=("at_bound", "mean"), mean_price_change=("price_change_pct", "mean")
    )
    save_csv(
        aggregate.merge(boundaries, on=keys),
        output,
        "pricing_sensitivity_aggregate.csv",
    )
    save_csv(uncertainty, output, "pricing_uncertainty_store.csv")
    aggregate_uncertainty = []
    for scenario, (gains, base) in aggregate_draws.items():
        aggregate_uncertainty.append(
            dict(
                zip(keys, scenario),
                baseline_profit=base,
                gain_p05=np.quantile(gains, 0.05),
                gain_p50=np.median(gains),
                gain_p95=np.quantile(gains, 0.95),
                positive_draw_share=(gains > 0).mean(),
                draws=settings["draws"],
            )
        )
    save_csv(aggregate_uncertainty, output, "pricing_uncertainty_aggregate.csv")
    save_csv(support_records, output, "relative_price_constraints.csv")
    save_csv(derivatives, output, "profit_derivatives.csv")
    full_panel_derivatives(panel, models, output)
    robust_screen(panel, models, draws, references, history, weights, settings, output)
    search_audit(models, references, history, weights, settings, output)
    # Compatibility projections come from this one pipeline, not separate estimators/searches.
    primary = store_results.loc[
        store_results.system.eq("linked_product")
        & store_results.specification.eq("week_FE")
        & store_results.maxchange.eq(max(settings["price_changes"]))
    ].copy()
    primary["cost_scenario"] = primary.cost_multiplier.map(
        {0.9: "cost_minus10", 1: "observed_cost", 1.1: "cost_plus10"}
    )
    primary["percent_gain"] = np.where(
        primary.baseline_profit.gt(0),
        100 * (primary.profit / primary.baseline_profit - 1),
        np.nan,
    )
    save_csv(
        primary[
            [
                "store",
                "week",
                "cost_scenario",
                "policy",
                "profit",
                "percent_gain",
                "converged",
            ]
        ],
        output,
        "pricing_scenarios.csv",
    )
    primary_details = product_results.loc[
        product_results.system.eq("linked_product")
        & product_results.specification.eq("week_FE")
        & product_results.maxchange.eq(max(settings["price_changes"]))
    ].copy()
    primary_details["cost_scenario"] = primary_details.cost_multiplier.map(
        {0.9: "cost_minus10", 1: "observed_cost", 1.1: "cost_plus10"}
    )
    primary_details = primary_details.rename(columns={"cost": "accounting_cost"})
    save_csv(
        primary_details[
            [
                "store",
                "week",
                "product",
                "cost_scenario",
                "policy",
                "price",
                "quantity",
                "accounting_cost",
                "lower_bound",
                "upper_bound",
                "at_bound",
            ]
        ],
        output,
        "pricing_product_details.csv",
    )
    save_json(
        dict(
            reference_week=week,
            reference_stores=len(references),
            main_exercise="linked_product / pair_linked",
            unrestricted_robustness="product / product_free",
            retained_linked_comparison="product / pair_linked; same unrestricted elasticities with linked decision constraints",
            profit="accounting gross contribution; excludes operating costs",
            dynamic_simulation="persistent proportional change of current and all four lagged prices; not a dynamic control solution",
            history_max_week=week,
            baseline_support_exceptions_allowed=True,
            uncertainty="joint two-way clustered Gaussian coefficient draws; baseline levels/weights/cost paths fixed; no sign truncation",
            optimized_scope="four-product portfolio; omitted products' profits and traffic effects excluded",
            search_guarantee="linked finite grid exhaustive; unrestricted block coordinate is local, audited against global grid examples",
        ),
        output / "pricing_notes.json",
    )


def search_audit(models, references, history, weights, settings, output):
    """Compare fitted models at references using history/weights and settings.
    Write independent finite-grid/block-search comparisons to output; return no data."""
    rows = []
    for key in [
        ("linked_product", "week_FE"),
        ("product", "week_FE"),
        ("product", "season_trend"),
    ]:
        model = models[key]
        index = references.index[0]
        row = references.iloc[0]
        baseline_prices, lo, hi, ratios, _ = support_for_store(
            history, row, 0.15, settings["support_quantiles"]
        )
        elasticities, baseline_units = model_inputs(model, index, row, weights)
        cost = row[[f"cost{j}" for j in range(1, 5)]].to_numpy(float)
        linked = key[0] != "product"
        _, solution, _, method, _ = optimize(
            baseline_prices,
            baseline_units,
            cost,
            elasticities,
            lo,
            hi,
            ratios,
            linked,
            settings["grid_points"],
            settings["maximum_iterations"],
        )
        if linked:
            choices = linked_grid(
                baseline_prices, lo, hi, ratios, settings["grid_points"]
            )
        else:
            grids = [
                np.unique(
                    np.r_[
                        np.linspace(lo[j], hi[j], settings["grid_points"]),
                        baseline_prices[j],
                    ]
                )
                for j in range(4)
            ]
            mesh = np.meshgrid(*grids, indexing="ij")
            choices = np.column_stack([a.ravel() for a in mesh])
            choices = choices[feasible(choices, lo, hi, ratios)]
        global_value = profit(
            choices, baseline_prices, baseline_units, cost, elasticities
        ).max()
        chosen_value = profit(
            solution, baseline_prices, baseline_units, cost, elasticities
        )[0]
        dense = linked_grid(
            baseline_prices, lo, hi, ratios, 2 * settings["grid_points"] - 1
        )
        rows.append(
            dict(
                system=key[0],
                specification=key[1],
                store=int(row.store),
                method=method,
                finite_grid_candidates=len(choices),
                chosen_profit=chosen_value,
                global_finite_grid_profit=global_value,
                local_global_gap=global_value - chosen_value,
                dense_linked_grid_profit=profit(
                    dense, baseline_prices, baseline_units, cost, elasticities
                ).max(),
                guarantee="this audit store only; continuous global optimum not established",
            )
        )
    save_csv(rows, output, "search_audit.csv")


def price_one_store(
    key,
    model,
    rule,
    index,
    row,
    change,
    multiplier,
    history,
    weights,
    settings,
    draws,
    aggregate_draws,
):
    """Return policy/profit/uncertainty records for one store and scenario.

    Inputs retain the same fitted baseline, common coefficient draws and
    historical support across policies. Returned lists use existing schemas.
    """
    linked = rule == "pair_linked"
    week = int(row.week)
    summaries, details, uncertainty, support_records, derivatives = [], [], [], [], []
    baseline_prices, lower, upper, ratios, exceptions = support_for_store(
        history, row, change, settings["support_quantiles"]
    )
    costs = row[[f"cost{j}" for j in range(1, 5)]].to_numpy(float) * multiplier
    elasticities, baseline_units = model_inputs(model, index, row, weights)
    independent, joint, converged, method, evaluations = optimize(
        baseline_prices,
        baseline_units,
        costs,
        elasticities,
        lower,
        upper,
        ratios,
        linked,
        settings["grid_points"],
        settings["maximum_iterations"],
    )
    metadata = dict(
        system=key[0],
        specification=key[1],
        price_rule=rule,
        maxchange=change,
        cost_multiplier=multiplier,
        store=int(row.store),
        week=week,
    )
    base = float(np.sum((baseline_prices - costs) * baseline_units))
    for policy, prices in [
        ("observed", baseline_prices),
        ("independent", independent),
        ("joint", joint),
    ]:
        assert feasible(prices, lower, upper, ratios)[0]
        if linked:
            assert (
                np.max(
                    np.abs(
                        prices[[0, 2]] / baseline_prices[[0, 2]]
                        - prices[[1, 3]] / baseline_prices[[1, 3]]
                    )
                )
                < 1e-8
            )
        quantities = predict_units(
            prices, baseline_prices, baseline_units, elasticities
        )[0]
        value = float(np.sum((prices - costs) * quantities))
        summaries.append(
            dict(
                metadata,
                policy=policy,
                profit=value,
                baseline_profit=base,
                converged=converged if policy == "joint" else np.nan,
                method=method,
                independent_unit="family" if linked else "product",
                independent_definition="hold rivals at baseline; common-ray shrink for feasible simultaneous rollout",
                candidate_evaluations=evaluations,
            )
        )
        for j in range(4):
            details.append(
                dict(
                    metadata,
                    policy=policy,
                    product=j + 1,
                    price=prices[j],
                    quantity=quantities[j],
                    cost=costs[j],
                    baseline_price=baseline_prices[j],
                    lower_bound=lower[j],
                    upper_bound=upper[j],
                    price_change_pct=100 * (prices[j] / baseline_prices[j] - 1),
                    at_bound=int(
                        abs(prices[j] - lower[j]) < 1e-8
                        or abs(prices[j] - upper[j]) < 1e-8
                    ),
                )
            )
        if policy == "observed":
            continue
        gains = drawn_gains(prices, baseline_prices, baseline_units, costs, draws[key])[
            :, 0
        ]
        uncertainty.append(
            dict(
                metadata,
                policy=policy,
                point_gain=value - base,
                gain_p05=np.quantile(gains, 0.05),
                gain_p50=np.median(gains),
                gain_p95=np.quantile(gains, 0.95),
                positive_draw_share=(gains > 0).mean(),
                conditional_baseline=True,
            )
        )
        scenario = (key[0], key[1], rule, change, multiplier, policy)
        if scenario not in aggregate_draws:
            aggregate_draws[scenario] = [np.zeros(settings["draws"]), 0.0]
        aggregate_draws[scenario][0] += gains
        aggregate_draws[scenario][1] += base
    if (
        multiplier == 1
        and change == settings["price_changes"][-1]
        and not (model.system == "product" and linked)
    ):
        derivative = profit_gradient(
            baseline_prices, baseline_units, costs, elasticities
        )
        for j in range(4):
            derivatives.append(
                dict(
                    system=key[0],
                    specification=key[1],
                    store=int(row.store),
                    product=j + 1,
                    joint_profit_derivative=derivative[j],
                    own_elasticity=elasticities[j, j],
                    margin=(baseline_prices[j] - costs[j]) / baseline_prices[j],
                )
            )
    if key == ("linked_product", "week_FE") and multiplier == 1:
        for k, (a, b) in enumerate(PAIRS):
            support_records.append(
                dict(
                    store=int(row.store),
                    week=week,
                    maxchange=change,
                    product_a=a + 1,
                    product_b=b + 1,
                    lower_log_ratio=ratios[k, 0],
                    upper_log_ratio=ratios[k, 1],
                    baseline_exceptions=exceptions,
                )
            )
    return summaries, details, uncertainty, support_records, derivatives


def price_one_model(
    key, model, rule, references, history, weights, settings, draws, aggregate_draws
):
    """Execute all cost/cap scenarios for a named demand system and price rule."""
    summaries, details, uncertainty, support_records, derivatives = [], [], [], [], []
    for change, multiplier in product(
        settings["price_changes"], settings["cost_multipliers"]
    ):
        for index, row in references.iterrows():
            (
                store_rows,
                product_rows,
                uncertainty_rows,
                support_rows,
                derivative_rows,
            ) = price_one_store(
                key,
                model,
                rule,
                index,
                row,
                change,
                multiplier,
                history,
                weights,
                settings,
                draws,
                aggregate_draws,
            )
            summaries.extend(store_rows)
            details.extend(product_rows)
            uncertainty.extend(uncertainty_rows)
            support_records.extend(support_rows)
            derivatives.extend(derivative_rows)
    return summaries, details, uncertainty, support_records, derivatives
