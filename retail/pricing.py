"""Historical support, finite-grid search and paired parameter-uncertainty stress screens.

Inputs: reference prices/quantities, response slopes, costs and historical bounds.
Outputs: demand predictions, gross contributions, feasible masks and searched policies.
"""

from itertools import combinations
import logging
import numpy as np

LOG = logging.getLogger(__name__)
PAIRS = list(combinations(range(4), 2))


def predict_units(prices, baseline_prices, baseline_units, elasticities, form="loglog"):
    """Predict mean package sales anchored at reference prices and quantities.

    prices is a candidate-by-product array (or one price vector). Reference
    prices/units have four entries; elasticities has outcome-by-price slopes.
    Return a candidate-by-product mean-sales array for the requested form.

    Log1p slopes act on q+1. Semilog slopes multiply dollar price changes;
    the default log-log form multiplies log price changes. No signs are imposed.
    """
    prices = np.atleast_2d(prices)
    assert (prices > 0).all()
    changes = (
        prices - baseline_prices
        if form == "semilog"
        else np.log(prices / baseline_prices)
    )
    factor = np.exp(changes @ elasticities.T)
    return (
        (baseline_units + 1) * factor - 1
        if form == "log1p"
        else baseline_units * factor
    )


def profit(prices, baseline_prices, baseline_units, costs, elasticities, form="loglog"):
    """Return one full-portfolio gross contribution per candidate price vector.

    Use reference prices/units, product costs and outcome-by-price slopes.
    Negative predicted quantities yield an infeasible (-infinity) objective.
    """
    units = predict_units(prices, baseline_prices, baseline_units, elasticities, form)
    values = ((np.atleast_2d(prices) - costs) * units).sum(axis=1)
    return np.where((units < 0).any(axis=1), -np.inf, values)


def feasible(prices, lower, upper, ratio_bounds, tolerance=1e-8):
    """Return a candidate mask for marginal-price boxes and every historical pairwise log-ratio constraint."""
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
    """Return reference prices, supported boxes, ratio ranges and explicit baseline exceptions using that store history."""
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
    """Return all feasible proportional Pepsi/Coke multiplier pairs on the configured finite grid."""
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
    form="loglog",
):
    """Choose each own-profit decision with rivals held at baseline, then return an order-invariant feasible simultaneous rollout."""
    planned = baseline_prices.copy()
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
            choices = np.tile(baseline_prices, (len(grid), 1))
            choices[:, group] = baseline_prices[group] * grid[:, None]
        else:
            grid = np.unique(
                np.r_[
                    np.linspace(lower[group[0]], upper[group[0]], points),
                    baseline_prices[group[0]],
                ]
            )
            choices = np.tile(baseline_prices, (len(grid), 1))
            choices[:, group[0]] = grid
        choices = choices[feasible(choices, lower, upper, ratios)]
        own = (
            (choices[:, group] - cost[group])
            * predict_units(
                choices, baseline_prices, baseline_units, elasticities, form
            )[:, group]
        ).sum(axis=1)
        planned[group] = choices[np.argmax(own), group]
    # Each own-profit choice holds rivals at their observed price. Their
    # simultaneous rollout can violate a shared price-ratio constraint.
    # Shrink the assembled log changes along a common ray until feasible;
    # this is order invariant and its full-system profit is evaluated below.
    log_change = np.log(planned / baseline_prices)
    fraction = 1.0
    for k, (a, b) in enumerate(PAIRS):
        slope = log_change[a] - log_change[b]
        base_ratio = np.log(baseline_prices[a] / baseline_prices[b])
        if slope > 1e-14:
            fraction = min(fraction, (ratios[k, 1] - base_ratio) / slope)
        elif slope < -1e-14:
            fraction = min(fraction, (ratios[k, 0] - base_ratio) / slope)
    rollout = baseline_prices * np.exp(max(0.0, fraction) * log_change)
    assert feasible(rollout, lower, upper, ratios)[0]
    return rollout


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
    form="loglog",
):
    """Search policies using reference prices/units, costs and price-response slopes.

    lower/upper are product price bounds; ratios holds six pairwise log-ranges.
    linked restricts proportional brand decisions. points controls the finite
    grids and maximum_iterations limits block passes. Return independent and
    joint policies, convergence, a search label and an evaluation count.
    The linked grid is exhaustive; unrestricted block search is local.
    """
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
        form,
    )
    linked_options = linked_grid(baseline_prices, lower, upper, ratios, points)
    assert len(linked_options) > 0
    linked_best = linked_options[
        np.argmax(
            profit(
                linked_options,
                baseline_prices,
                baseline_units,
                cost,
                elasticities,
                form,
            )
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
    # Numerical search nests starts, passes and two-product blocks; each level
    # has a distinct convergence/search purpose.
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
    best_value = profit(
        best, baseline_prices, baseline_units, cost, elasticities, form
    )[0]
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
                    choices, baseline_prices, baseline_units, cost, elasticities, form
                )
                evaluations += len(choices)
                chosen = np.argmax(values)
                if (
                    values[chosen]
                    > profit(
                        current,
                        baseline_prices,
                        baseline_units,
                        cost,
                        elasticities,
                        form,
                    )[0]
                    + 1e-9
                ):
                    current = choices[chosen]
                    changed = True
            if not changed:
                finished = True
                break
        value = profit(
            current, baseline_prices, baseline_units, cost, elasticities, form
        )[0]
        if value > best_value + 1e-9:
            best, best_value, converged = current, value, int(finished)
        elif abs(value - best_value) <= 1e-9 and finished:
            converged = 1
    assert (
        best_value
        >= profit(
            independent, baseline_prices, baseline_units, cost, elasticities, form
        )[0]
        - 1e-6
    )
    assert (
        best_value
        >= profit(
            linked_best, baseline_prices, baseline_units, cost, elasticities, form
        )[0]
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
    """Map product, linked-product or family slopes/draws to the four-price decision space using frozen family weights."""
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
    """Return a four-price slope matrix and conditional baseline quantities for one reference row; family shares allocate observed mix."""
    elasticities = expand_elasticities(model.elasticity, model.system, weights)
    baseline_units = model.baseline_units[index].copy()
    if model.system == "family":
        observed = row[[f"move{j}" for j in range(1, 5)]].to_numpy(float)
        shares = observed.reshape(2, 2) / observed.reshape(2, 2).sum(axis=1)[:, None]
        baseline_units = (baseline_units[:, None] * shares).ravel()
    return elasticities, baseline_units


def common_reference(panel, models, minimum_stores):
    """Return the latest week and stores with costs and fitted baselines jointly usable in every primary model."""
    mask = panel[[f"cost{j}" for j in range(1, 5)]].notna().all(axis=1).to_numpy()
    for model in models.values():
        mask &= np.isfinite(model.baseline_units).all(axis=1)
    counts = panel.loc[mask].groupby("week").store.nunique()
    week = int(counts.loc[counts.ge(minimum_stores)].index.max())
    return week, panel.loc[mask & panel.week.eq(week).to_numpy()]


def profit_gradient(prices, units, costs, elasticities):
    """Each price affects every product's margin contribution through cross responses."""
    return units + (((prices - costs) * units) @ elasticities) / prices
