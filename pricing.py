"""Historical support, finite-grid search and paired parameter-uncertainty stress screens."""

from itertools import combinations, product
import logging
import numpy as np
import pandas as pd
from .io import save_csv, save_json

LOG = logging.getLogger(__name__)
PAIRS = list(combinations(range(4), 2))


def predict_units(prices, baseline_prices, baseline_units, elasticities):
    return baseline_units * np.exp(
        np.log(np.atleast_2d(prices) / baseline_prices) @ elasticities.T
    )


def profit(prices, baseline_prices, baseline_units, costs, elasticities):
    return np.sum(
        (np.atleast_2d(prices) - costs)
        * predict_units(prices, baseline_prices, baseline_units, elasticities),
        axis=1,
    )


def feasible(prices, lower, upper, ratio_bounds, tolerance=1e-8):
    prices = np.atleast_2d(prices)
    valid = (prices >= lower - tolerance).all(axis=1) & (
        prices <= upper + tolerance
    ).all(axis=1)
    for k, (a, b) in enumerate(PAIRS):
        ratios = np.log(prices[:, a] / prices[:, b])
        valid &= (ratios >= ratio_bounds[k, 0] - tolerance) & (
            ratios <= ratio_bounds[k, 1] + tolerance
        )
    return valid


def support_for_store(history, row, change, quantiles):
    baseline_prices = row[[f"p{j}" for j in range(1, 5)]].to_numpy(float)
    prices = history.loc[
        history.store.eq(row.store), [f"p{j}" for j in range(1, 5)]
    ].to_numpy(float)
    lo, hi = np.quantile(prices, quantiles, axis=0)
    lower = np.minimum(baseline_prices, np.maximum(baseline_prices * (1 - change), lo))
    upper = np.maximum(baseline_prices, np.minimum(baseline_prices * (1 + change), hi))
    bounds = []
    exceptions = 0
    for a, b in PAIRS:
        left, right = np.quantile(np.log(prices[:, a] / prices[:, b]), quantiles)
        base = np.log(baseline_prices[a] / baseline_prices[b])
        exceptions += int(base < left - 1e-8 or base > right + 1e-8)
        bounds.append([min(left, base), max(right, base)])
    return baseline_prices, lower, upper, np.array(bounds), exceptions


def linked_grid(baseline_prices, lower, upper, ratio_bounds, points):
    grids = []
    for pair in [(0, 1), (2, 3)]:
        left = np.max(lower[list(pair)] / baseline_prices[list(pair)])
        right = np.min(upper[list(pair)] / baseline_prices[list(pair)])
        grids.append(np.unique(np.r_[np.linspace(left, right, points), 1.0]))
    a, b = np.meshgrid(*grids, indexing="ij")
    prices = baseline_prices * np.column_stack(
        [a.ravel(), a.ravel(), b.ravel(), b.ravel()]
    )
    return prices[feasible(prices, lower, upper, ratio_bounds)]


def independent_policy(
    baseline_prices,
    baseline_units,
    cost,
    elasticities,
    lower,
    upper,
    ratios,
    linked,
    points,
):
    prices = baseline_prices.copy()
    groups = [(0, 1), (2, 3)] if linked else [(j,) for j in range(4)]
    for group in groups:
        group = np.array(group)
        if linked:
            grid = np.unique(
                np.r_[
                    np.linspace(
                        np.max(lower[group] / baseline_prices[group]),
                        np.min(upper[group] / baseline_prices[group]),
                        points,
                    ),
                    1,
                ]
            )
            choices = np.tile(prices, (len(grid), 1))
            choices[:, group] = baseline_prices[group] * grid[:, None]
        else:
            grid = np.unique(
                np.r_[
                    np.linspace(lower[group[0]], upper[group[0]], points),
                    baseline_prices[group[0]],
                ]
            )
            choices = np.tile(prices, (len(grid), 1))
            choices[:, group[0]] = grid
        choices = choices[feasible(choices, lower, upper, ratios)]
        own = (
            (choices[:, group] - cost[group])
            * predict_units(choices, baseline_prices, baseline_units, elasticities)[
                :, group
            ]
        ).sum(axis=1)
        prices = choices[np.argmax(own)]
    return prices


def optimize(
    baseline_prices,
    baseline_units,
    cost,
    elasticities,
    lower,
    upper,
    ratios,
    linked,
    points=21,
    maximum_iterations=100,
):
    independent = independent_policy(
        baseline_prices,
        baseline_units,
        cost,
        elasticities,
        lower,
        upper,
        ratios,
        linked,
        points,
    )
    linked_options = linked_grid(baseline_prices, lower, upper, ratios, points)
    assert len(linked_options) > 0
    linked_best = linked_options[
        np.argmax(
            profit(linked_options, baseline_prices, baseline_units, cost, elasticities)
        )
    ]
    if linked:
        return (
            independent,
            linked_best,
            1,
            "exhaustive_feasible_2D_grid",
            len(linked_options),
        )
    # Two-product blocks can move nearly equal regular/diet prices together.
    # Single-coordinate search alone can stall under tight observed relative-price support.
    grids = [
        np.unique(np.r_[np.linspace(lower[j], upper[j], points), baseline_prices[j]])
        for j in range(4)
    ]
    blocks = []
    for a, b in [(0, 1), (2, 3)]:
        first, second = np.meshgrid(grids[a], grids[b], indexing="ij")
        blocks.append(np.column_stack([first.ravel(), second.ravel()]))
    starts = [
        baseline_prices,
        independent,
        linked_best,
        lower,
        upper,
        (lower + upper) / 2,
    ]
    best = baseline_prices.copy()
    best_value = profit(best, baseline_prices, baseline_units, cost, elasticities)[0]
    converged = 0
    evaluations = 0
    for start in starts:
        if not feasible(start, lower, upper, ratios)[0]:
            continue
        current = start.copy()
        finished = False
        for _ in range(maximum_iterations):
            changed = False
            for group, grid in zip([(0, 1), (2, 3)], blocks):
                choices = np.tile(current, (len(grid), 1))
                choices[:, list(group)] = grid
                choices = choices[feasible(choices, lower, upper, ratios)]
                values = profit(
                    choices, baseline_prices, baseline_units, cost, elasticities
                )
                evaluations += len(choices)
                chosen = np.argmax(values)
                if (
                    values[chosen]
                    > profit(
                        current, baseline_prices, baseline_units, cost, elasticities
                    )[0]
                    + 1e-9
                ):
                    current = choices[chosen]
                    changed = True
            if not changed:
                finished = True
                break
        value = profit(current, baseline_prices, baseline_units, cost, elasticities)[0]
        if value > best_value + 1e-9:
            best, best_value, converged = current, value, int(finished)
        elif abs(value - best_value) <= 1e-9 and finished:
            converged = 1
    assert (
        best_value
        >= profit(independent, baseline_prices, baseline_units, cost, elasticities)[0]
        - 1e-6
    )
    assert (
        best_value
        >= profit(linked_best, baseline_prices, baseline_units, cost, elasticities)[0]
        - 1e-6
    )
    return (
        independent,
        best,
        converged,
        "multistart_block_coordinate_4D_grid",
        evaluations,
    )


def expand_elasticities(values, system, weights):
    values = np.asarray(values)
    if system == "product":
        return values
    if system == "family":
        values = np.repeat(values, 2, axis=-2)
    return np.stack(
        [
            values[..., 0] * weights[0],
            values[..., 0] * weights[1],
            values[..., 1] * weights[2],
            values[..., 1] * weights[3],
        ],
        axis=-1,
    )


def model_inputs(model, index, row, weights):
    elasticities = expand_elasticities(model.elasticity, model.system, weights)
    baseline_units = model.baseline_units[index].copy()
    if model.system == "family":
        observed = row[[f"move{j}" for j in range(1, 5)]].to_numpy(float)
        shares = observed.reshape(2, 2) / observed.reshape(2, 2).sum(axis=1)[:, None]
        baseline_units = (baseline_units[:, None] * shares).ravel()
    return elasticities, baseline_units


def parameter_draws(model, settings, rng, weights):
    values, vectors = np.linalg.eigh(model.price_covariance)
    clipped = np.maximum(values, 0)
    root = vectors * np.sqrt(clipped)[None, :]
    draws = (
        model.elasticity.ravel()
        + rng.normal(size=(settings["draws"], len(values))) @ root.T
    )
    draws = draws.reshape((settings["draws"],) + model.elasticity.shape)
    audit = dict(
        system=model.system,
        specification=model.specification,
        minimum_covariance_eigenvalue=values.min(),
        clipped_eigenvalues=int((values < 0).sum()),
        removed_negative_eigenvalue_mass=float(-values[values < 0].sum()),
        covariance_trace=float(values.sum()),
        draws=settings["draws"],
        nonnegative_own_draw_share=float(
            (
                np.diagonal(
                    expand_elasticities(draws, model.system, weights), axis1=1, axis2=2
                )
                >= 0
            ).mean()
        ),
    )
    return expand_elasticities(draws, model.system, weights), audit


def drawn_gains(choices, baseline_prices, baseline_units, cost, draws):
    log_changes = np.log(np.atleast_2d(choices) / baseline_prices)
    units = baseline_units * np.exp(
        np.einsum("cp,dqp->dcq", log_changes, draws, optimize=True)
    )
    return np.sum((np.atleast_2d(choices)[None, :, :] - cost) * units, axis=2) - np.sum(
        (baseline_prices - cost) * baseline_units
    )


def common_reference(panel, models, minimum_stores):
    mask = panel[[f"cost{j}" for j in range(1, 5)]].notna().all(axis=1).to_numpy()
    for model in models.values():
        mask &= np.isfinite(model.baseline_units).all(axis=1)
    counts = panel.loc[mask].groupby("week").store.nunique()
    week = int(counts.loc[counts.ge(minimum_stores)].index.max())
    return week, panel.loc[mask & panel.week.eq(week).to_numpy()]


def run_pricing(panel, models, config):
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
    for key, model in models.items():
        linked = model.system != "product"
        rule = "pair_linked" if linked else "product_free"
        LOG.info("Pricing %s / %s (%s stores)", *key, len(references))
        for change, multiplier in product(
            settings["price_changes"], settings["cost_multipliers"]
        ):
            for index, row in references.iterrows():
                baseline_prices, lower, upper, ratios, exceptions = support_for_store(
                    history, row, change, settings["support_quantiles"]
                )
                costs = (
                    row[[f"cost{j}" for j in range(1, 5)]].to_numpy(float) * multiplier
                )
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
                                price_change_pct=100
                                * (prices[j] / baseline_prices[j] - 1),
                                at_bound=int(
                                    abs(prices[j] - lower[j]) < 1e-8
                                    or abs(prices[j] - upper[j]) < 1e-8
                                ),
                            )
                        )
                    if policy == "observed":
                        continue
                    gains = drawn_gains(
                        prices, baseline_prices, baseline_units, costs, draws[key]
                    )[:, 0]
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
                if multiplier == 1 and change == settings["price_changes"][-1]:
                    derivative = (
                        baseline_units
                        + (
                            elasticities.T
                            @ ((baseline_prices - costs) * baseline_units)
                        )
                        / baseline_prices
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
                                margin=(baseline_prices[j] - costs[j])
                                / baseline_prices[j],
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
    store_results = save_csv(summaries, output, "pricing_sensitivity_store.csv")
    product_results = save_csv(details, output, "pricing_sensitivity_details.csv")
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
    robust_screen(panel, models, draws, references, history, weights, settings, output)
    search_audit(models, references, history, weights, settings, output)
    # Compatibility projections come from this one pipeline, not separate estimators/searches.
    primary = store_results.loc[
        store_results.system.eq("linked_product")
        & store_results.specification.eq("week_FE")
        & store_results.maxchange.eq(0.15)
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
        & product_results.maxchange.eq(0.15)
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


def robust_screen(panel, models, draws, references, history, weights, settings, output):
    # Screen the same linked policy against unrestricted-product, linked-product
    # and aggregate-family estimates, including omitted-substitute and lag models.
    selected = models
    rows, detail = [], []
    aggregate = {}
    for index, row in references.iterrows():
        baseline_prices, lower, upper, ratios, _ = support_for_store(
            history,
            row,
            settings["robust_maximum_change"],
            settings["support_quantiles"],
        )
        choices = linked_grid(
            baseline_prices, lower, upper, ratios, settings["robust_grid_points"]
        )
        worst = np.full(len(choices), np.inf)
        for key, model in selected.items():
            _, baseline_units = model_inputs(model, index, row, weights)
            for multiplier in settings["cost_multipliers"]:
                cost = (
                    row[[f"cost{j}" for j in range(1, 5)]].to_numpy(float) * multiplier
                )
                distribution = drawn_gains(
                    choices, baseline_prices, baseline_units, cost, draws[key]
                )
                score = np.quantile(
                    distribution, settings["lower_tail"], axis=0
                ) / np.sum(baseline_prices * baseline_units)
                worst = np.minimum(worst, score)
        best = int(np.argmax(worst))
        prices = choices[best]
        # Baseline is in every candidate set and has zero gain in every parameter draw.
        assert worst[best] >= -1e-10
        active = np.max(np.abs(prices / baseline_prices - 1)) > 1e-8
        rows.append(
            dict(
                store=int(row.store),
                week=int(row.week),
                feasible_candidates=len(choices),
                worst_p05_gain_fraction_of_revenue=worst[best],
                changes_prices=int(active),
                robust_screen_pass=int(worst[best] > 1e-8 and active),
                pepsi_change_pct=100 * (prices[0] / baseline_prices[0] - 1),
                coke_change_pct=100 * (prices[2] / baseline_prices[2] - 1),
            )
        )
        for j in range(4):
            detail.append(
                dict(
                    store=int(row.store),
                    week=int(row.week),
                    product=j + 1,
                    price=prices[j],
                    baseline_price=baseline_prices[j],
                )
            )
        for key, model in selected.items():
            elasticities, baseline_units = model_inputs(model, index, row, weights)
            for multiplier in settings["cost_multipliers"]:
                cost = (
                    row[[f"cost{j}" for j in range(1, 5)]].to_numpy(float) * multiplier
                )
                gains = drawn_gains(
                    prices, baseline_prices, baseline_units, cost, draws[key]
                )[:, 0]
                group = (key[0], key[1], multiplier)
                if group not in aggregate:
                    aggregate[group] = [np.zeros(settings["draws"]), 0, 0]
                aggregate[group][0] += gains
                aggregate[group][1] += np.sum((baseline_prices - cost) * baseline_units)
                aggregate[group][2] += profit(
                    prices, baseline_prices, baseline_units, cost, elasticities
                )[0] - np.sum((baseline_prices - cost) * baseline_units)
    save_csv(rows, output, "robust_price_recommendations.csv")
    save_csv(detail, output, "robust_price_details.csv")
    summary = []
    for key, (gains, base, point) in aggregate.items():
        summary.append(
            dict(
                system=key[0],
                specification=key[1],
                cost_multiplier=key[2],
                baseline_profit=base,
                point_gain=point,
                point_gain_pct=100 * point / base if base > 0 else np.nan,
                gain_p05=np.quantile(gains, 0.05),
                gain_p95=np.quantile(gains, 0.95),
                positive_draw_share=(gains > 0).mean(),
            )
        )
    save_csv(summary, output, "robust_policy_aggregate.csv")


def search_audit(models, references, history, weights, settings, output):
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
