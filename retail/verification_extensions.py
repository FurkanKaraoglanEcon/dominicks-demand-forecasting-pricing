"""Verify auxiliary demand, transformed objectives and operational chronology.

Inputs: computed tables, prediction files and fitted models from this run.
Outputs: checks appended to the common verification table and summary. Checks
recompute economic identities and selection rules instead of matching snapshots.
"""

import numpy as np
import pandas as pd
from .io import save_csv, save_json
from .forecast_operational import accuracy, feature_names, construct_forecast_data
from .pricing import predict_units, profit


def verify_forecast_outputs(config, check, close):
    """Check recorded origins, selection, level errors and retrospective labels."""
    output = config["paths"]["output"]
    validation = pd.read_parquet(output / "validation_predictions.parquet")
    test = pd.read_parquet(output / "test_predictions.parquet")
    fits = pd.read_csv(output / "tables/forecast_fit_inventory.csv")
    check(
        "operational and conditional fits precede every target",
        (test.train_through < test.week).all()
        and (validation.train_through < validation.week).all(),
    )
    check(
        "all operational evaluation labeled retrospective",
        test.evaluation.eq("retrospective").all(),
    )
    check(
        "intervals contain their point forecast",
        ((test.lower <= test.prediction) & (test.upper >= test.prediction)).all(),
    )
    check("nonnegative sales forecasts", test.prediction.ge(0).all())
    operational = fits[fits.information_set.str.startswith("operational")]
    check(
        "operational regressors exclude realized current policy",
        not operational.feature_columns.str.contains("current").any(),
    )
    columns, _ = feature_names("operational")
    check(
        "operational price and promotion covariates explicitly lagged",
        all("lag1" in c for c in columns if "price" in c or "promotion" in c),
    )
    calibrated = fits[fits.point_calibration_last.notna()]
    check(
        "point calibration uses only prior held-out errors",
        (
            (
                calibrated.point_calibration_prefix_last
                < calibrated.point_calibration_first
            )
            & (calibrated.point_calibration_last <= calibrated.train_through)
        ).all(),
    )
    selection = pd.read_csv(output / "tables/forecast_selection.csv")
    scores = pd.read_csv(output / "tables/forecast_validation_scores.csv")
    for row in selection.itertuples():
        sample = scores[
            (scores.category == row.category)
            & (scores.information_set == row.information_set)
        ]
        minimum = sample.sort_values("WAPE", kind="stable").iloc[0]
        check(
            f"{row.category} {row.information_set}: validation winner frozen",
            row.selected_model == minimum.model_id,
        )
        check(
            f"{row.category} {row.information_set}: selection before calibration and test",
            row.selection_last_week < row.calibration_last_week < row.test_first_week,
        )
    metrics = pd.read_csv(output / "tables/forecast_test_metrics.csv")
    for row in metrics.itertuples():
        sample = test[
            (test.category == row.category)
            & (test.information_set == row.information_set)
            & (test.model_id == row.model_id)
        ]
        independent = accuracy(sample.target, sample.prediction)
        close(
            f"{row.category} {row.information_set} {row.model_id}: WAPE identity",
            row.WAPE,
            independent["WAPE"],
        )
        close(
            f"{row.category} {row.information_set} {row.model_id}: interval coverage",
            row.interval_coverage,
            sample.covered.mean(),
        )
    # Contaminate unavailable current/future prices, promotions and outcomes;
    # Permitted features at the first available test origin must remain unchanged.
    raw = pd.read_parquet(config["paths"]["processed"] / "panel_sdr.parquet")
    before = construct_forecast_data(raw)
    test_weeks = before.loc[before.week.gt(config["sample"]["calibration_end"]), "week"]
    assert len(test_weeks) > 0, "Operational verification needs a held-out target week"
    origin = int(test_weeks.min())
    changed = raw.copy()
    future = changed.week.ge(origin)
    changed.loc[future, "units_p1"] *= 100
    changed.loc[future, "price_p1"] *= 100
    changed.loc[future, "promotion_p1"] = 1 - changed.loc[future, "promotion_p1"]
    after = construct_forecast_data(changed)
    left = before[before.week.le(origin)][columns].reset_index(drop=True)
    right = after[after.week.le(origin)][columns].reset_index(drop=True)
    check("operational future outcome/price/promotion perturbation", left.equals(right))


def verify_form_outputs(config, check, close):
    """Reconcile accounting profit and verify finite-grid dominance and support."""
    output = config["paths"]["output"]
    details = pd.read_csv(output / "pricing/functional_form_details.csv")
    results = pd.read_csv(output / "pricing/functional_form_scenarios.csv")
    keys = ["category", "form", "maximum_change", "cost_multiplier", "policy"]
    computed = (
        details.assign(contribution=(details.price - details.cost) * details.quantity)
        .groupby(keys)
        .contribution.sum()
    )
    close(
        "functional-form gross contribution identity",
        results.set_index(keys).profit.sort_index(),
        computed.sort_index(),
    )
    check(
        "functional-form prices in supported boxes",
        (
            (details.price >= details.lower_bound - 1e-8)
            & (details.price <= details.upper_bound + 1e-8)
        ).all(),
    )
    check("functional-form quantities positive", details.quantity.gt(0).all())
    bootstrap = pd.read_csv(output / "pricing/store_wild_draws.csv")
    check(
        "store bootstrap reoptimized grid includes fixed policy",
        (bootstrap.reoptimized_grid_gain >= bootstrap.fixed_policy_gain - 1e-6).all(),
    )
    check(
        "store bootstrap scenarios executed",
        len(bootstrap)
        == len(config["categories"]) * config["pricing"]["store_wild_draws"],
    )
    audit = pd.read_csv(output / "pricing/functional_form_search_audit.csv")
    check(
        "functional-form search dominates exhaustive finite grid",
        (audit.chosen_profit >= audit.grid_profit - 1e-6).all(),
    )
    check(
        "all categories and transformed objectives executed",
        len(audit)
        == len(config["categories"])
        * 3
        * len(config["pricing"]["form_price_changes"])
        * len(config["pricing"]["cost_multipliers"]),
    )
    # Derivatives have an independent finite-difference benchmark for each form.
    p0 = np.array([2.0, 3.0, 4.0, 5.0])
    q0 = np.array([100.0, 80.0, 60.0, 40.0])
    cost = np.ones(4)
    beta = np.diag([-2.0, -3.0, -2.0, -4.0]) + 0.1
    for form in ["loglog", "log1p", "semilog"]:
        quantity = predict_units(p0, p0, q0, beta, form)[0]
        level = quantity + 1 if form == "log1p" else quantity
        jacobian = level[:, None] * beta
        if form != "semilog":
            jacobian /= p0[None, :]
        derivative = quantity + (p0 - cost) @ jacobian
        numeric = []
        for j in range(4):
            plus, minus = p0.copy(), p0.copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            numeric.append(
                (
                    profit(plus, p0, q0, cost, beta, form)[0]
                    - profit(minus, p0, q0, cost, beta, form)[0]
                )
                / 2e-5
            )
        close(f"{form}: analytic cross-profit derivatives", derivative, numeric, 1e-5)


def verify_auxiliary_outputs(config, check, close):
    """Check executed inference/sample controls, ranks and matched comparisons."""
    output = config["paths"]["output"]
    coefficients = pd.read_csv(output / "tables/demand_coefficients.csv")
    expected = {
        "loglog",
        "log1p",
        "semilog",
        "log1p_common_positive",
        "income_interactions",
        "OLS_income_sample",
        "OLS_visit_sample",
        "per_visit",
        "traffic_control",
        "static_dynamic_sample",
        "dynamic",
        "AAC_IV_sensitivity",
        "OLS_IV_sample",
        "category_expansion",
    }
    for category in config["categories"]:
        executed = set(coefficients[coefficients.category == category].model)
        check(
            f"{category}: every auxiliary demand exercise executed",
            expected.issubset(executed),
        )
        group = coefficients[coefficients.category == category]
        check(
            f"{category}: finite auxiliary standard errors",
            np.isfinite(group.standard_error).all(),
        )
        for a, b in [
            ("income_interactions", "OLS_income_sample"),
            ("per_visit", "OLS_visit_sample"),
            ("traffic_control", "OLS_visit_sample"),
            ("AAC_IV_sensitivity", "OLS_IV_sample"),
            ("dynamic", "static_dynamic_sample"),
        ]:
            close(
                f"{category}: matched samples {a}/{b}",
                group[group.model == a].groupby("outcome").n.first().sort_index(),
                group[group.model == b].groupby("outcome").n.first().sort_index(),
            )
    cola = coefficients[
        (coefficients.category == "sdr")
        & (coefficients.model == "loglog")
        & coefficients.term.str.startswith("price_")
    ]
    primary = pd.read_csv(output / "demand/demand_sensitivity.csv")
    primary = primary[
        (primary.system == "product") & (primary.specification == "week_FE")
    ].sort_values(["demand", "price"])
    close(
        "shared auxiliary/primary OLS same-sample point estimates",
        cola.sort_values(["outcome", "term"]).coefficient,
        primary.elasticity,
        1e-8,
    )
    close(
        "shared auxiliary/primary OLS same-sample inference",
        cola.sort_values(["outcome", "term"]).standard_error,
        primary.se,
        1e-8,
    )
    for category in ["cer", "lnd"]:
        ranks = pd.read_csv(output / f"tables/product_candidates_{category}.csv")
        chosen = (
            ranks[ranks.eligible]
            .sort_values(["revenue", "upc"], ascending=[False, True])
            .head(4)
            .upc
        )
        fixed_upcs = config["categories"][category].get("fixed_upcs", [])
        if fixed_upcs:
            chosen = fixed_upcs
            check(
                f"{category}: fixed benchmark training eligibility",
                ranks.set_index("upc").loc[chosen, "eligible"].all(),
            )
        mapping = pd.read_csv(output / "tables/product_mapping.csv")
        check(
            f"{category}: training-only deterministic ranks",
            set(chosen) == set(mapping[mapping.category == category].upc),
        )


def verify_extensions(config, summary):
    """Append independent extension checks and update the common verification report."""
    output = config["paths"]["output"]
    rows = []

    def check(name, condition, detail=""):
        rows.append(
            dict(
                check=name,
                status="PASS" if bool(condition) else "FAIL",
                detail=str(detail),
            )
        )

    def close(name, actual, expected, atol=1e-6):
        actual = np.asarray(actual, float)
        expected = np.asarray(expected, float)
        check(
            name,
            np.allclose(actual, expected, atol=atol, rtol=1e-8, equal_nan=True),
            f"absolute tolerance {atol:g}; relative tolerance 1e-8",
        )

    verify_auxiliary_outputs(config, check, close)
    verify_form_outputs(config, check, close)
    verify_forecast_outputs(config, check, close)
    old = pd.read_csv(output / "verification_checks.csv")
    checks = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    save_csv(checks, output, "verification_checks.csv")
    summary.update(
        checks=len(checks),
        passed=int(checks.status.eq("PASS").sum()),
        failed=int(checks.status.eq("FAIL").sum()),
        warnings=int(checks.status.eq("WARN").sum()),
    )
    save_json(summary, output / "verification_report.json")
    return summary
