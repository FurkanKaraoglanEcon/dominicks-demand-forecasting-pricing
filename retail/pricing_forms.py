"""Functional-form and additional-portfolio robustness with the shared price objective.

Inputs: auxiliary fitted equations, prepared panels and the primary robust policy.
Outputs: supported representative-market simulations for all three portfolios,
finite-grid/search audits and a common-policy stress table. Main recommendations
remain store-specific linked cola prices; these extensions are conditional checks.
"""

from itertools import product
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .io import save_csv, save_json
from .econometrics import fit_system
from .product_definitions import PRODUCTS, PRICE_NAMES, PROMOTION_NAMES
from .pricing import predict_units, profit, feasible, optimize, PAIRS


def reference_form(panel, form, config):
    """Fit a joint equation system and anchor at training medians and smeared means."""
    data = panel.copy()
    positive = data[[f"units_{p}" for p in PRODUCTS]].gt(0).all(axis=1)
    sample = np.ones(len(data), bool) if form == "log1p" else positive.to_numpy()
    regressors = PRICE_NAMES + PROMOTION_NAMES
    for p in PRODUCTS:
        quantity = data[f"units_{p}"]
        data[f"outcome_{p}"] = (
            np.log1p(quantity)
            if form == "log1p"
            else np.log(quantity.where(quantity.gt(0)))
        )
        if form != "semilog":
            data[f"price_{p}"] = np.log(data[f"price_{p}"])
    fit = fit_system(data, [f"outcome_{p}" for p in PRODUCTS], regressors, sample, True)
    selected = panel.loc[fit.mask]
    smeared = (
        np.exp(fit.fitted)
        * pd.DataFrame(np.exp(fit.residual))
        .groupby(selected.store.to_numpy())
        .transform("mean")
        .to_numpy()
    )
    mean_units = smeared - 1 if form == "log1p" else smeared
    assert (mean_units > 0).all()
    end = config["sample"]["train_end"]
    anchor = selected.week.between(end - 51, end)
    assert anchor.any()
    p0 = selected.loc[anchor, PRICE_NAMES].median().to_numpy()
    costs = selected.loc[anchor, [f"cost_{p}" for p in PRODUCTS]].median().to_numpy()
    quantity = mean_units[anchor].mean(axis=0)
    # Anchor every equation at the same representative prices, averaging
    # fitted means adjusted from the row's actual prices to that reference.
    observed_prices = selected.loc[anchor, PRICE_NAMES].to_numpy()
    change = p0 - observed_prices if form == "semilog" else np.log(p0 / observed_prices)
    beta = fit.beta[:4].T
    factor = np.exp(change @ beta.T)
    quantity = (
        ((mean_units[anchor] + 1) * factor - 1).mean(axis=0)
        if form == "log1p"
        else (mean_units[anchor] * factor).mean(axis=0)
    )
    prices = panel.loc[panel.week.le(end), PRICE_NAMES].to_numpy()
    quantiles = config["pricing"]["support_quantiles"]
    support = np.quantile(prices, quantiles, axis=0)
    ratios = np.array(
        [np.quantile(np.log(prices[:, a] / prices[:, b]), quantiles) for a, b in PAIRS]
    )
    base_ratios = np.array([np.log(p0[a] / p0[b]) for a, b in PAIRS])
    exceptions = ((base_ratios < ratios[:, 0]) | (base_ratios > ratios[:, 1])).sum()
    ratios[:, 0] = np.minimum(ratios[:, 0], base_ratios)
    ratios[:, 1] = np.maximum(ratios[:, 1], base_ratios)
    indices = np.array([e * len(regressors) + j for e in range(4) for j in range(4)])
    covariance = fit.covariance[np.ix_(indices, indices)]
    return p0, quantity, costs, beta, support, ratios, covariance, int(exceptions), fit


def global_grid(lower, upper, p0, ratios, points):
    """Enumerate and filter the finite four-price grid, including the baseline."""
    axes = [
        np.unique(np.r_[np.linspace(lo, hi, points), base])
        for lo, hi, base in zip(lower, upper, p0)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    choices = np.column_stack([coordinate.ravel() for coordinate in mesh])
    return choices[feasible(choices, lower, upper, ratios)]


def continuous_refinement(start, p0, q0, cost, beta, lower, upper, ratios, form):
    """Refine a feasible grid candidate with explicit log-ratio inequalities.

    The finite grid has a global guarantee over its enumerated choices only.
    SLSQP improves precision locally; it does not certify a continuous optimum.
    """
    constraints = []
    for k, (a, b) in enumerate(PAIRS):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda p, a=a, b=b, k=k: np.log(p[a] / p[b]) - ratios[k, 0],
            }
        )
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda p, a=a, b=b, k=k: ratios[k, 1] - np.log(p[a] / p[b]),
            }
        )
    result = minimize(
        lambda p: -profit(p, p0, q0, cost, beta, form)[0],
        start,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 200},
    )
    accepted = (
        result.success
        and feasible(result.x, lower, upper, ratios, 1e-8)[0]
        and profit(result.x, p0, q0, cost, beta, form)[0]
        >= profit(start, p0, q0, cost, beta, form)[0] - 1e-6
    )
    return result.x if accepted else start, dict(
        success=bool(result.success),
        accepted=bool(accepted),
        message=str(result.message),
    )


def restricted_policies(p0, q0, cost, beta, lower, upper, ratios, form, points):
    """Evaluate common proportional and equal-absolute brand pricing sensitivities."""
    left = np.max(lower / p0)
    right = np.min(upper / p0)
    choices = p0 * np.unique(np.r_[np.linspace(left, right, points), 1])[:, None]
    choices = choices[feasible(choices, lower, upper, ratios)]
    policies = {
        "all_products_common_multiplier": choices[
            np.argmax(profit(choices, p0, q0, cost, beta, form))
        ]
    }
    axes = [
        np.linspace(max(lower[a], lower[b]), min(upper[a], upper[b]), points)
        for a, b in [(0, 1), (2, 3)]
    ]
    first, second = np.meshgrid(*axes, indexing="ij")
    absolute = np.column_stack(
        [first.ravel(), first.ravel(), second.ravel(), second.ravel()]
    )
    absolute = absolute[feasible(absolute, lower, upper, ratios)]
    if len(absolute):
        policies["brand_equal_absolute_price"] = absolute[
            np.argmax(profit(absolute, p0, q0, cost, beta, form))
        ]
    return policies, len(absolute)


def representative_scenario(category, form, reference, change, multiplier, config):
    """Search one supported scenario; return policy details and a numerical audit."""
    p0, q0, cost, beta, support, ratios, covariance, exceptions, fit = reference
    cost = cost * multiplier
    lower = np.minimum(p0, np.maximum(p0 * (1 - change), support[0]))
    upper = np.maximum(p0, np.minimum(p0 * (1 + change), support[1]))
    # Nonnegative unit margin is a sensitivity restriction. Retain observed
    # loss leaders at baseline rather than silently discarding that reference.
    lower = np.minimum(p0, np.maximum(lower, cost))
    points = config["pricing"]["form_grid_points"]
    independent, joint, _, _, _ = optimize(
        p0, q0, cost, beta, lower, upper, ratios, False, points, 100, form
    )
    choices = global_grid(lower, upper, p0, ratios, points)
    grid = choices[np.argmax(profit(choices, p0, q0, cost, beta, form))]
    if (
        profit(grid, p0, q0, cost, beta, form)[0]
        > profit(joint, p0, q0, cost, beta, form)[0]
    ):
        joint = grid
    joint, solver = continuous_refinement(
        joint, p0, q0, cost, beta, lower, upper, ratios, form
    )
    _, linked, _, _, _ = optimize(
        p0, q0, cost, beta, lower, upper, ratios, True, points, 100, form
    )
    extra, absolute_choices = restricted_policies(
        p0, q0, cost, beta, lower, upper, ratios, form, points
    )
    policies = {
        "observed": p0,
        "independent": independent,
        "joint": joint,
        "pair_linked": linked,
        **extra,
    }
    base = profit(p0, p0, q0, cost, beta, form)[0]
    summaries, details = [], []
    for policy, prices in policies.items():
        assert feasible(prices, lower, upper, ratios)[0]
        quantity = predict_units(prices, p0, q0, beta, form)[0]
        value = profit(prices, p0, q0, cost, beta, form)[0]
        summaries.append(
            dict(
                category=category,
                form=form,
                maximum_change=change,
                cost_multiplier=multiplier,
                policy=policy,
                profit=value,
                baseline_profit=base,
                gain_pct=100 * (value / base - 1) if base > 0 else np.nan,
                scope="representative four-product portfolio",
            )
        )
        details.extend(
            dict(
                category=category,
                form=form,
                maximum_change=change,
                cost_multiplier=multiplier,
                policy=policy,
                product=p,
                price=prices[j],
                quantity=quantity[j],
                cost=cost[j],
                baseline_price=p0[j],
                lower_bound=lower[j],
                upper_bound=upper[j],
            )
            for j, p in enumerate(PRODUCTS)
        )
    audit = dict(
        category=category,
        form=form,
        maximum_change=change,
        cost_multiplier=multiplier,
        finite_grid_candidates=len(choices),
        grid_profit=profit(grid, p0, q0, cost, beta, form)[0],
        chosen_profit=profit(joint, p0, q0, cost, beta, form)[0],
        support_baseline_exceptions=exceptions,
        equal_absolute_candidates=absolute_choices,
        **solver,
    )
    assert audit["chosen_profit"] >= audit["grid_profit"] - 1e-6
    return summaries, details, audit


def common_policy_forms(cola, config):
    """Stress the primary store policy under log1p/semilog and cost assumptions.

    This evaluates an unchanged policy; each form is not allowed to substitute
    its own optimum. Cross-equation two-way draws match the main uncertainty.
    """
    robust = pd.read_csv(config["paths"]["output"] / "pricing/robust_price_details.csv")
    results = []
    rng = np.random.default_rng(config["pricing"]["seed"] + 1)
    for form in ["loglog", "log1p", "semilog"]:
        data = cola.copy()
        if form == "semilog":
            prices = [f"p{j}" for j in range(1, 5)]
        else:
            prices = [f"lp{j}" for j in range(1, 5)]
        outcomes = [f"zq{j}" if form == "log1p" else f"lq{j}" for j in range(1, 5)]
        regressors = prices + [f"promo_flag{j}" for j in range(1, 5)]
        fit = fit_system(data, outcomes, regressors, np.ones(len(data), bool), True)
        q = np.full((len(data), 4), np.nan)
        q[fit.mask] = np.exp(fit.fitted) * np.exp(fit.residual).mean(axis=0) - (
            1 if form == "log1p" else 0
        )
        beta = fit.beta[:4].T
        indices = np.array([e * 8 + j for e in range(4) for j in range(4)])
        covariance = fit.covariance[np.ix_(indices, indices)]
        values, vectors = np.linalg.eigh(covariance)
        draws = (
            beta.ravel()
            + rng.normal(size=(config["pricing"]["draws"], 16))
            @ (vectors * np.sqrt(np.maximum(values, 0))).T
        )
        draws = draws.reshape(-1, 4, 4)
        for multiplier in config["pricing"]["cost_multipliers"]:
            results.append(
                evaluate_common_form_policy(
                    data, robust, q, beta, draws, multiplier, form, values
                )
            )
    save_csv(
        results,
        config["paths"]["output"] / "pricing",
        "common_policy_functional_forms.csv",
    )


def run_form_pricing(cola, models, config):
    """Execute all functional-form, cap and cost scenarios from prepared inputs."""
    summaries, details, audits = [], [], []
    for category, form in product(config["categories"], ["loglog", "log1p", "semilog"]):
        panel = pd.read_parquet(
            config["paths"]["processed"] / f"panel_{category}.parquet"
        )
        reference = reference_form(panel, form, config)
        for change, multiplier in product(
            config["pricing"]["form_price_changes"],
            config["pricing"]["cost_multipliers"],
        ):
            summary, detail, audit = representative_scenario(
                category, form, reference, change, multiplier, config
            )
            summaries.extend(summary)
            details.extend(detail)
            audits.append(audit)
    directory = config["paths"]["output"] / "pricing"
    save_csv(summaries, directory, "functional_form_scenarios.csv")
    save_csv(details, directory, "functional_form_details.csv")
    save_csv(audits, directory, "functional_form_search_audit.csv")
    common_policy_forms(cola, config)
    save_json(
        dict(
            reference="medians and adjusted fitted means in final 52 operational-training weeks",
            history_max_week=config["sample"]["train_end"],
            scope="additional representative portfolio simulations; not directly comparable with store-aggregate primary pricing",
            relative_support="all six pairwise log-price-ratio training quantiles; explicit baseline exceptions",
        ),
        directory / "functional_form_notes.json",
    )


def evaluate_common_form_policy(data, robust, q, beta, draws, multiplier, form, values):
    """Aggregate fixed-policy gains and draw distributions for one form/cost case."""
    aggregate = np.zeros(len(draws))
    point = 0
    negative_draws = 0
    for store, group in robust.groupby("store"):
        row = data[(data.store == store) & (data.week == group.week.iloc[0])].iloc[0]
        p0 = row[[f"p{j}" for j in range(1, 5)]].to_numpy(float)
        q0 = q[row.name]
        cost = row[[f"cost{j}" for j in range(1, 5)]].to_numpy(float) * multiplier
        prices = group.sort_values("product").price.to_numpy()
        change = prices - p0 if form == "semilog" else np.log(prices / p0)
        quantity = (
            (q0 + 1) * np.exp(np.einsum("p,dqp->dq", change, draws)) - 1
            if form == "log1p"
            else q0 * np.exp(np.einsum("p,dqp->dq", change, draws))
        )
        invalid = (quantity < 0).any(axis=1)
        negative_draws += invalid.sum()
        gains = ((prices - cost) * quantity).sum(axis=1) - ((p0 - cost) * q0).sum()
        gains[invalid] = -np.inf
        aggregate += gains
        point += (
            profit(prices, p0, q0, cost, beta, form)[0]
            - profit(p0, p0, q0, cost, beta, form)[0]
        )
    return dict(
        form=form,
        cost_multiplier=multiplier,
        point_gain=point,
        gain_p05=np.quantile(aggregate, 0.05),
        gain_p95=np.quantile(aggregate, 0.95),
        positive_draw_share=(aggregate > 0).mean(),
        negative_quantity_store_draws=negative_draws,
        clipped_covariance_eigenvalues=int((values < 0).sum()),
        interpretation="fixed primary policy; conditional baseline; two-way joint draws",
    )
