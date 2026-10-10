"""Joint two-way coefficient draws and common linked-policy stress screens.

Inputs: clustered covariances and conditional reference demand/costs.
Outputs: unrestricted-sign draws, covariance audits and robust policy records.
Draws are shared across stores; this does not include baseline or cost uncertainty.
"""

from itertools import product
import numpy as np
from .io import save_csv
from .pricing import (
    expand_elasticities,
    model_inputs,
    support_for_store,
    linked_grid,
    profit,
)


def parameter_draws(model, settings, rng, weights):
    """Draw joint slopes from model covariance using settings and rng.
    Expand family slopes with weights; return product-level draws and an eigenvalue audit.
    """
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
    """Evaluate choices relative to baseline_prices/baseline_units with cost and draws.
    Return a draw-by-candidate matrix of full-portfolio gross-contribution gains."""
    log_changes = np.log(np.atleast_2d(choices) / baseline_prices)
    units = baseline_units * np.exp(
        np.einsum("cp,dqp->dcq", log_changes, draws, optimize=True)
    )
    return np.sum((np.atleast_2d(choices)[None, :, :] - cost) * units, axis=2) - np.sum(
        (baseline_prices - cost) * baseline_units
    )


def robust_screen(panel, models, draws, references, history, weights, settings, output):
    # Screen the same linked policy against unrestricted-product, linked-product
    # and aggregate-family estimates, including omitted-substitute and lag models.
    """Search common linked policies using models/draws at references.
    Use panel history, weights and settings; write lower-tail store/policy results to output.
    """
    selected = models
    model_cost_cases = list(product(selected.items(), settings["cost_multipliers"]))
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
        for (key, model), multiplier in model_cost_cases:
            elasticities, baseline_units = model_inputs(model, index, row, weights)
            cost = row[[f"cost{j}" for j in range(1, 5)]].to_numpy(float) * multiplier
            distribution = drawn_gains(
                choices, baseline_prices, baseline_units, cost, draws[key]
            )
            score = np.quantile(distribution, settings["lower_tail"], axis=0) / np.sum(
                baseline_prices * baseline_units
            )
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
        for (key, model), multiplier in model_cost_cases:
            elasticities, baseline_units = model_inputs(model, index, row, weights)
            cost = row[[f"cost{j}" for j in range(1, 5)]].to_numpy(float) * multiplier
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
