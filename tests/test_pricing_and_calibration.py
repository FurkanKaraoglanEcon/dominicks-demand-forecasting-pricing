"""Known optima, order-invariant own pricing and finite-sample calibration ranks."""

import numpy as np
from retail.pricing import independent_policy, profit, optimize, feasible, predict_units
from retail.pricing_forms import continuous_refinement
from retail.econometrics import fixed_effect_rank
from retail.forecast_operational import interval_radius


def test_independent_decisions_hold_rivals_fixed_and_are_order_invariant():
    p0 = np.ones(4)
    q0 = np.full(4, 100.0)
    cost = np.full(4, 0.5)
    beta = np.diag([-3.0, -1.5, -2.0, -2.0])
    ratios = np.tile([np.log(0.98), -np.log(0.98)], (6, 1))
    lower, upper = p0 * 0.8, p0 * 1.2
    original = independent_policy(p0, q0, cost, beta, lower, upper, ratios, False, 21)
    permutation = np.array([1, 0, 2, 3])
    permuted = independent_policy(
        p0[permutation],
        q0[permutation],
        cost[permutation],
        beta[np.ix_(permutation, permutation)],
        lower[permutation],
        upper[permutation],
        ratios,
        False,
        21,
    )
    restored = permuted[np.argsort(permutation)]
    np.testing.assert_allclose(original, restored, atol=1e-12, rtol=1e-12)
    assert feasible(original, lower, upper, ratios)[0]


def test_known_interior_profit_optimum_with_explicit_tolerance():
    p0 = np.array([2.0, 3.0, 4.0, 5.0])
    q0 = np.array([100.0, 80.0, 60.0, 40.0])
    cost = np.array([1.0, 1.5, 2.0, 2.5])
    beta = np.diag([-2.0, -3.0, -2.0, -4.0])
    lower = p0 * 0.5
    upper = p0 * 2
    ratios = np.tile([-10.0, 10.0], (6, 1))
    _, grid, _, _, _ = optimize(p0, q0, cost, beta, lower, upper, ratios, False, 21)
    optimum, record = continuous_refinement(
        grid, p0, q0, cost, beta, lower, upper, ratios, "loglog"
    )
    known = np.diag(beta) * cost / (1 + np.diag(beta))
    np.testing.assert_allclose(optimum, known, atol=1e-5, rtol=1e-6)
    assert record["accepted"]
    np.testing.assert_allclose(
        profit(optimum, p0, q0, cost, beta),
        ((optimum - cost) * predict_units(optimum, p0, q0, beta)[0]).sum(),
        atol=1e-10,
    )


def test_interval_uses_requested_order_statistic():
    scores = np.arange(1.0, 101.0)
    assert interval_radius(scores, 0.9) == 91.0
    assert interval_radius(np.array([2.0, 4.0]), 0.9) == 4.0


def test_disconnected_fixed_effect_graph_rank():
    store = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    week = np.array([1, 2, 1, 2, 3, 4, 3, 4])
    assert fixed_effect_rank(store, week) == 6
