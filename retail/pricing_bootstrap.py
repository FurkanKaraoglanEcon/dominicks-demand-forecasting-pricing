"""Store-only wild-score sensitivity beside the main two-way uncertainty.

Inputs: the same auxiliary log-log equations and representative reference.
Outputs: fixed-policy and reoptimized finite-grid gains for shared store draws.
This preserves the uploaded exercise, but it omits common week dependence and
therefore does not replace the main two-way recommendation screen.
"""

import pickle
import numpy as np
import pandas as pd
from .econometrics import group_sum
from .io import save_csv
from .product_definitions import PRODUCTS, PRICE_NAMES
from .pricing_forms import reference_form, global_grid


def store_score_draws(fits, draws, rng):
    """Perturb each equation with the same store-level Rademacher multipliers."""
    first = fits["p1"]
    stores, codes = np.unique(first["store"], return_inverse=True)
    multipliers = rng.choice([-1.0, 1.0], size=(draws, len(stores)))
    coefficients = np.empty((draws, 4, 4))
    diagnostics = []
    for equation, product in enumerate(PRODUCTS):
        fit = fits[product]
        assert np.array_equal(fit["store"], first["store"])
        scores = group_sum(fit["x_within"] * fit["residuals"][:, None], codes)
        parameters = fit["coefficients"] + multipliers @ scores @ fit["bread"]
        indices = [fit["names"].index(name) for name in PRICE_NAMES]
        coefficients[:, equation, :] = parameters[:, indices]
        diagnostics.append(
            dict(
                product=product,
                store_only_draw_own_sd=parameters[:, indices[equation]].std(ddof=1),
                two_way_own_se=np.sqrt(
                    fit["covariance"][indices[equation], indices[equation]]
                ),
            )
        )
    return coefficients, diagnostics


def bootstrap_gains(choices, policy, p0, q0, cost, draws):
    """Evaluate fixed policy and maximum feasible finite-grid gain per draw."""
    base = ((p0 - cost) * q0).sum()
    units = q0 * np.exp(np.einsum("p,dqp->dq", np.log(policy / p0), draws))
    fixed = ((policy - cost) * units).sum(axis=1) - base
    best = np.full(len(draws), -np.inf)
    # Numerical batches bound memory while preserving every grid candidate.
    for first in range(0, len(choices), 128):
        candidate = choices[first : first + 128]
        quantity = q0 * np.exp(np.einsum("cp,dqp->dcq", np.log(candidate / p0), draws))
        gains = ((candidate[None, :, :] - cost) * quantity).sum(axis=2) - base
        best = np.maximum(best, gains.max(axis=1))
    return fixed, best


def run_store_wild_sensitivity(config):
    """Execute the uploaded store-only bootstrap on each category's main form."""
    rng = np.random.default_rng(config["pricing"]["seed"] + 2)
    output = config["paths"]["output"]
    rows = []
    diagnostics = []
    summaries = []
    policies = pd.read_csv(output / "pricing/functional_form_details.csv")
    cap = max(config["pricing"]["form_price_changes"])
    for category in config["categories"]:
        with (output / "models" / f"demand_{category}.pkl").open("rb") as stream:
            fits = pickle.load(stream)["loglog"]
        panel = pd.read_parquet(
            config["paths"]["processed"] / f"panel_{category}.parquet"
        )
        p0, q0, cost, beta, support, ratios, _, _, _ = reference_form(
            panel, "loglog", config
        )
        lower = np.minimum(p0, np.maximum.reduce([p0 * (1 - cap), support[0], cost]))
        upper = np.maximum(p0, np.minimum(p0 * (1 + cap), support[1]))
        policy = (
            policies[
                (policies.category == category)
                & (policies.form == "loglog")
                & (policies.cost_multiplier == 1)
                & (policies.maximum_change == cap)
                & (policies.policy == "joint")
            ]
            .sort_values("product")
            .price.to_numpy()
        )
        choices = np.vstack(
            [
                global_grid(
                    lower, upper, p0, ratios, config["pricing"]["form_grid_points"]
                ),
                policy,
            ]
        )
        draws, audit = store_score_draws(
            fits, config["pricing"]["store_wild_draws"], rng
        )
        fixed, reoptimized = bootstrap_gains(choices, policy, p0, q0, cost, draws)
        rows.extend(
            dict(
                category=category,
                draw=j + 1,
                fixed_policy_gain=a,
                reoptimized_grid_gain=b,
            )
            for j, (a, b) in enumerate(zip(fixed, reoptimized))
        )
        diagnostics.extend(dict(category=category, **record) for record in audit)
        summaries.append(
            dict(
                category=category,
                draws=len(draws),
                fixed_gain_p05=np.quantile(fixed, 0.05),
                fixed_gain_p95=np.quantile(fixed, 0.95),
                reoptimized_gain_p05=np.quantile(reoptimized, 0.05),
                reoptimized_gain_p95=np.quantile(reoptimized, 0.95),
                omitted_dependence="common week shocks; sensitivity only",
            )
        )
    save_csv(rows, output / "pricing", "store_wild_draws.csv")
    save_csv(summaries, output / "pricing", "store_wild_summary.csv")
    save_csv(diagnostics, output / "pricing", "store_wild_vs_two_way.csv")
