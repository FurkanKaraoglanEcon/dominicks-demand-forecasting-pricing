"""Economic and chronology regressions, rather than snapshots of implementation internals."""

import unittest
import numpy as np
import pandas as pd
from retail.io import calendar_lag
from retail.preparation import valid_movement
from retail.econometrics import forecast_levels
from retail.pricing import predict_units, profit, feasible, optimize, profit_gradient
from retail.identification import event_windows
from retail.verification_research import dummy_ols_check


class ResearchRules(unittest.TestCase):
    def test_bundled_sales_and_invalid_prices(self):
        frame = pd.DataFrame(
            dict(
                store=[2, 2],
                upc=[1, 1],
                week=[1, 2],
                move=[9, 0],
                qty=[3, 1],
                price=[2.0, 0.0],
                ok=[1, 1],
            )
        )
        valid = valid_movement(frame, {"first_week": 1, "last_week": 399})
        self.assertEqual(valid.tolist(), [True, False])
        self.assertEqual(
            (
                frame.loc[valid, "price"]
                / frame.loc[valid, "qty"]
                * frame.loc[valid, "move"]
            ).iloc[0],
            6,
        )

    def test_lags_do_not_jump_gaps(self):
        frame = pd.DataFrame(
            dict(store=[2, 2, 2], week=[1, 3, 4], q=[10.0, 20.0, 30.0])
        )
        lag = calendar_lag(frame, "q", 1)
        self.assertTrue(np.isnan(lag[:2]).all())
        self.assertEqual(lag[2], 20)

    def test_cross_responses_enter_portfolio_profit(self):
        p0 = np.array([1.0, 1.0, 1.0, 1.0])
        q0 = np.array([10.0, 20.0, 30.0, 40.0])
        cost = np.full(4, 0.7)
        B = np.diag([-2.0, -2.0, -2.0, -2.0])
        B[2, 0] = 0.8
        B[3, 1] = -0.6
        prices = p0.copy()
        prices[0] = 1.1
        expected = q0 * np.exp(B @ np.log(prices / p0))
        np.testing.assert_allclose(predict_units(prices, p0, q0, B)[0], expected)
        self.assertGreater(expected[2], q0[2])
        self.assertAlmostEqual(
            profit(prices, p0, q0, cost, B)[0], ((prices - cost) * expected).sum()
        )
        analytic = profit_gradient(p0, q0, cost, B)
        numerical = []
        for j in range(4):
            plus, minus = p0.copy(), p0.copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            numerical.append(
                (profit(plus, p0, q0, cost, B)[0] - profit(minus, p0, q0, cost, B)[0])
                / 2e-5
            )
        np.testing.assert_allclose(analytic, numerical, atol=1e-6, rtol=1e-8)

    def test_event_windows_require_calendar_and_stable_controls(self):
        panel = pd.DataFrame(
            dict(
                store=np.full(9, 2),
                week=np.arange(1, 10),
                fq1=[100.0] * 4 + [110.0] * 5,
                fq2=np.full(9, 80.0),
                flp2=np.zeros(9),
            )
        )
        for j in range(1, 5):
            panel[f"promo_flag{j}"] = 0
        anchors = pd.DataFrame(dict(store=[2], week=[5]))
        events = panel.week.eq(5).to_numpy()
        rows, complete, clean = event_windows(
            panel, anchors, "flp1", ["flp2"], events, 4, 0.01
        )
        self.assertEqual((complete, clean, len(rows)), (1, 1, 9))
        np.testing.assert_allclose(
            rows.focal_family_pct, [0.0] * 4 + [10.0] * 5, atol=1e-10
        )
        changed = panel.copy()
        changed.loc[changed.week.eq(6), "promo_flag1"] = 1
        _, complete, clean = event_windows(
            changed, anchors, "flp1", ["flp2"], events, 4, 0.01
        )
        self.assertEqual((complete, clean), (1, 0))
        overlapping = events | panel.week.eq(7).to_numpy()
        _, complete, clean = event_windows(
            panel, anchors, "flp1", ["flp2"], overlapping, 4, 0.01
        )
        self.assertEqual((complete, clean), (1, 0))
        gap = panel.loc[panel.week.ne(3)].copy()
        rows, complete, clean = event_windows(
            gap, anchors, "flp1", ["flp2"], gap.week.eq(5).to_numpy(), 4, 0.01
        )
        self.assertEqual((complete, clean, len(rows)), (0, 0, 0))

    def test_independent_dummy_ols(self):
        slope, covariance = dummy_ols_check()
        self.assertLess(slope, 1e-9)
        self.assertLess(covariance, 1e-9)

    def test_retransformation_is_level_mean(self):
        rng = np.random.default_rng(12)
        frame = pd.DataFrame(
            [(s, w) for s in range(5) for w in range(40)], columns=["store", "week"]
        )
        frame["x"] = rng.normal(size=len(frame))
        frame["y"] = (
            2 + 0.3 * frame.x + 0.1 * frame.store + rng.normal(size=len(frame)) * 0.7
        )
        sample = frame.week.lt(30).to_numpy()
        target = frame.week.eq(30).to_numpy()
        predicted, fit, smear = forecast_levels(frame, "y", ["x"], sample, target)
        np.testing.assert_allclose(smear, np.exp(fit.residual).mean())
        self.assertGreater(smear, 1)
        self.assertTrue((predicted >= 0).all())
        altered = frame.copy()
        altered.loc[altered.week.ge(30), "y"] = 10000
        revised, _, _ = forecast_levels(altered, "y", ["x"], sample, target)
        np.testing.assert_allclose(predicted, revised)

    def test_linkage_and_relative_support_are_simultaneous(self):
        p0 = np.ones(4)
        q0 = np.array([40.0, 30.0, 40.0, 30.0])
        cost = np.full(4, 0.8)
        B = np.diag(np.full(4, -2.0))
        lower = p0 * 0.9
        upper = p0 * 1.1
        ratios = np.array([[-0.02, 0.02]] * 6)
        independent, joint, _, _, _ = optimize(
            p0, q0, cost, B, lower, upper, ratios, True, 11, 50
        )
        self.assertTrue(feasible(joint, lower, upper, ratios)[0])
        self.assertAlmostEqual(joint[0], joint[1])
        self.assertAlmostEqual(joint[2], joint[3])
        self.assertGreaterEqual(
            profit(joint, p0, q0, cost, B)[0], profit(p0, p0, q0, cost, B)[0] - 1e-6
        )


if __name__ == "__main__":
    unittest.main()
