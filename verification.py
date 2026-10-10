"""Checks of economic identities, independent OLS algebra and chronological information sets."""

import json
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .io import save_csv, save_json, calendar_lag
from .econometrics import fit_system, forecast_levels
from .forecasting import features_for, calibrate_origins
from .pricing import support_for_store, feasible, model_inputs, profit


def dummy_ols_check():
    """Compare within slopes/covariance with a full dummy regression on an unbalanced panel."""
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        [(s, w) for s in range(7) for w in range(12) if (s + 2 * w) % 9 != 0],
        columns=["store", "week"],
    )
    frame["x"] = rng.normal(size=len(frame))
    frame["z"] = rng.normal(size=len(frame))
    frame["y"] = (
        1.2 * frame.x
        - 0.7 * frame.z
        + 0.2 * frame.store
        + 0.3 * frame.week
        + rng.normal(size=len(frame))
    )
    result = fit_system(frame, ["y"], ["x", "z"], np.ones(len(frame), bool), True)
    dummies = np.column_stack(
        [
            np.ones(len(frame)),
            pd.get_dummies(frame.store, drop_first=True, dtype=float),
            pd.get_dummies(frame.week, drop_first=True, dtype=float),
            frame[["x", "z"]],
        ]
    )
    beta = np.linalg.lstsq(dummies, frame.y, rcond=None)[0]
    residual = frame.y - dummies @ beta
    bread = np.linalg.inv(dummies.T @ dummies)
    n, k = dummies.shape

    def clustered(groups):
        sums = (
            pd.DataFrame(dummies * residual.to_numpy()[:, None])
            .groupby(groups.to_numpy())
            .sum()
            .to_numpy()
        )
        g = len(sums)
        return bread @ (sums.T @ sums) @ bread * g / (g - 1) * (n - 1) / (n - k)

    hc = (
        bread
        @ (
            (dummies * residual.to_numpy()[:, None]).T
            @ (dummies * residual.to_numpy()[:, None])
        )
        @ bread
        * n
        / (n - k)
    )
    covariance = clustered(frame.store) + clustered(frame.week) - hc
    return np.max(np.abs(result.beta[:, 0] - beta[-2:])), np.max(
        np.abs(result.covariance - covariance[-2:, -2:])
    )


def verify(config, panel, selected, models):
    output = config["paths"]["output"]
    settings = config["verification"]
    rows = []

    def check(name, condition, detail="", warning=False):
        rows.append(
            dict(
                check=name,
                status="PASS" if condition else ("WARN" if warning else "FAIL"),
                detail=str(detail),
            )
        )

    def close(name, actual, expected, atol=None):
        actual, expected = np.asarray(actual, float), np.asarray(expected, float)
        difference = np.nanmax(np.abs(actual - expected)) if actual.size else 0
        condition = np.allclose(
            actual,
            expected,
            atol=settings["absolute_tolerance"] if atol is None else atol,
            rtol=settings["relative_tolerance"],
            equal_nan=True,
        )
        check(name, condition, f"maximum absolute difference {difference:.6g}")

    category = pd.read_csv(output / "categories/category_summary.csv")
    for _, row in category.iterrows():
        check(
            f"{row.category}: complete raw accounting",
            row.raw_rows == row.valid_rows + row.invalid_rows,
            f"{int(row.raw_rows):,} rows read",
        )
        check(f"{row.category}: UPC metadata coverage", row.unknown_metadata_rows == 0)
        check(f"{row.category}: unique valid keys", row.duplicate_valid_keys == 0)
        products = pd.read_csv(
            output / f"categories/{row.category}_products_by_split.csv"
        )
        weekly = pd.read_csv(output / f"categories/{row.category}_store_week.csv")
        close(
            f"{row.category}: independent revenue aggregation",
            products.revenue.sum(),
            weekly.revenue.sum(),
        )
        close(
            f"{row.category}: observations aggregation",
            products.observations.sum(),
            row.valid_rows,
        )
    check("panel: unique store-week", not panel.duplicated(["store", "week"]).any())
    check(
        "panel: actual records for all four products",
        panel[[f"p{j}" for j in range(1, 5)] + [f"move{j}" for j in range(1, 5)]]
        .notna()
        .all()
        .all(),
    )
    check(
        "panel: training-only store-history restriction",
        panel.loc[panel.split.eq(1)].groupby("store").size().min()
        >= config["data"]["minimum_training_weeks"],
    )
    check(
        "panel: minimum stores",
        panel.store.nunique() >= config["data"]["minimum_stores"],
    )
    audit = json.loads((output / "preparation_audit.json").read_text())
    check(
        "calendar splits",
        panel.loc[panel.split.eq(1), "week"].max() == 279
        and panel.loc[panel.split.eq(2), "week"].max() == 339,
    )
    check(
        "source sample vs previous run",
        len(panel) == 31378 and panel.store.nunique() == 86,
        "Prior sample preserved; broader market changes coverage denominator",
        warning=True,
    )
    candidates = pd.read_csv(output / "candidate_training_coverage.csv")
    ranked = (
        candidates.loc[candidates.coverage.ge(config["data"]["minimum_coverage"])]
        .sort_values(["train_revenue", "upc"], ascending=[False, True])
        .head(4)
    )
    check(
        "product choice: deterministic training ranking",
        sorted(ranked.upc) == sorted(selected.upc),
    )
    full_products = pd.read_csv(output / "categories/sdr_products_by_split.csv")
    raw_train = full_products.loc[full_products.split.eq(1)].set_index("upc")
    close(
        "selection revenues: full raw training totals",
        candidates.train_revenue,
        raw_train.loc[candidates.upc, "revenue"],
    )
    close(
        "selection observations: full raw training totals",
        candidates.observed_weeks,
        raw_train.loc[candidates.upc, "observations"],
    )
    altered = full_products.copy()
    altered.loc[altered.split.ne(1), "revenue"] *= 1e9
    qualified = set(
        candidates.loc[
            candidates.coverage.ge(config["data"]["minimum_coverage"]), "upc"
        ]
    )
    reranked = (
        altered.loc[altered.split.eq(1) & altered.upc.isin(qualified)]
        .sort_values(["revenue", "upc"], ascending=[False, True])
        .head(4)
    )
    check(
        "selection: future-revenue perturbation",
        sorted(reranked.upc) == sorted(selected.upc),
    )
    check(
        "selection: last information week",
        audit["selection_last_week"] == audit["train_end"],
    )
    weights = (
        pd.read_csv(output / "family_weights.csv")
        .sort_values("product")
        .weight.to_numpy()
    )
    for family, pair in enumerate([(1, 2), (3, 4)], 1):
        revenue = np.array(
            [
                (
                    panel.loc[panel.split.eq(1), f"move{j}"]
                    * panel.loc[panel.split.eq(1), f"p{j}"]
                ).sum()
                for j in pair
            ]
        )
        close(
            f"family {family}: training-only weights",
            weights[2 * (family - 1) : 2 * family],
            revenue / revenue.sum(),
        )
    omitted = pd.read_csv(output / "omitted_substitute_weights.csv")
    close("omitted basket weights sum", omitted.weight.sum(), 1)
    close(
        "omitted basket training revenues",
        omitted.train_revenue,
        raw_train.loc[omitted.upc, "revenue"],
    )
    coverage = pd.read_csv(output / "revenue_coverage.csv")
    for _, row in coverage.iterrows():
        frame = full_products.loc[full_products.split.eq(row.split)]
        close(
            f"coverage {int(row.split)}: selected numerator",
            row.selected_revenue,
            frame.loc[frame.upc.isin(selected.upc), "revenue"].sum(),
        )
        close(
            f"coverage {int(row.split)}: broad denominator",
            row.market_2l_cola_revenue,
            frame.loc[frame.market_2l_cola, "revenue"].sum(),
        )
        close(
            f"coverage {int(row.split)}: percentage",
            row.market_coverage,
            row.selected_revenue / row.market_2l_cola_revenue,
        )
    for j in range(1, 5):
        finite = panel[f"cost{j}"].notna()
        close(
            f"product {j}: accounting cost",
            panel.loc[finite, f"cost{j}"],
            panel.loc[finite, f"p{j}"] * (1 - panel.loc[finite, f"margin{j}"]),
        )
        close(
            f"product {j}: calendar lag 1",
            panel[f"lag1_{j}"],
            calendar_lag(panel, f"zq{j}", 1),
        )
        close(
            f"product {j}: calendar lag 52",
            panel[f"lag52_{j}"],
            calendar_lag(panel, f"move{j}", 52),
        )
    traffic = pd.read_csv(output / "weekly_traffic.csv")
    check(
        "customer counts: seven positive valid days",
        traffic.loc[traffic.traffic.notna(), "traffic_days"].eq(7).all(),
    )
    check(
        "customer counts: household tracking not claimed",
        not json.loads((output / "customer_count_audit.json").read_text())[
            "household_ids_available"
        ],
    )
    slope_error, cov_error = dummy_ols_check()
    close("within OLS vs independent full dummy regression", slope_error, 0, 1e-9)
    close("two-way covariance vs independent full dummy regression", cov_error, 0, 1e-9)
    store_ids, week_ids = pd.factorize(panel.store)[0], pd.factorize(panel.week)[0]
    ns, nw = panel.store.nunique(), panel.week.nunique()
    graph = coo_matrix(
        (
            np.ones(2 * len(panel)),
            (np.r_[store_ids, week_ids + ns], np.r_[week_ids + ns, store_ids]),
        ),
        shape=(ns + nw, ns + nw),
    )
    check(
        "FE sample: connected store-week graph",
        connected_components(graph, directed=False)[0] == 1,
    )
    for key, model in models.items():
        fit = model.fit
        name = " / ".join(key)
        check(
            f"{name}: full column rank",
            np.linalg.matrix_rank(fit.within_x) == len(fit.regressors),
        )
        close(
            f"{name}: OLS orthogonality",
            fit.within_x.T @ fit.residual,
            np.zeros((len(fit.regressors), len(fit.outcomes))),
            1e-7,
        )
        close(
            f"{name}: fitted retransformation",
            model.baseline_units[fit.mask],
            np.exp(fit.fitted) * np.exp(fit.residual).mean(axis=0),
        )
        close(f"{name}: symmetric covariance", fit.covariance, fit.covariance.T, 1e-10)
        check(f"{name}: cluster counts", fit.stores > 1 and fit.weeks > 1)
        if key[1] in ["week_FE", "omitted_index", "distributed_lag"]:
            data = pd.DataFrame(fit.residual)
            close(
                f"{name}: store residual means",
                data.groupby(panel.loc[fit.mask, "store"].to_numpy()).mean(),
                0,
                1e-8,
            )
            close(
                f"{name}: week residual means",
                data.groupby(panel.loc[fit.mask, "week"].to_numpy()).mean(),
                0,
                1e-8,
            )
    dynamics = pd.read_csv(output / "demand/demand_dynamics.csv").pivot(
        index=["system", "demand", "price"], columns="response", values="elasticity"
    )
    close(
        "dynamic cumulative = immediate + subsequent",
        dynamics.cumulative,
        dynamics.immediate + dynamics.subsequent,
    )
    origins = pd.read_csv(output / "forecast/forecast_origin_audit.csv")
    rolling = origins.loc[origins.origin.gt(0)]
    check(
        "forecast rolling information sets",
        rolling.fit_last_week.lt(rolling.origin).all(),
    )
    check(
        "forecast rolling window bounds",
        rolling.fit_first_week.ge(
            rolling.origin - config["forecast"]["rolling_window"]
        ).all(),
    )
    fixed = origins.loc[origins.origin.eq(0)]
    check(
        "forecast fixed models: training only",
        fixed.fit_last_week.le(audit["train_end"]).all(),
    )
    factors = pd.read_csv(output / "forecast/forecast_calibration_factors.csv")
    check(
        "calibration: earlier out-of-sample errors only",
        factors.calibration_max_week.dropna()
        .lt(factors.loc[factors.calibration_max_week.notna(), "origin"])
        .all(),
    )
    check(
        "calibration: prespecified factor limits",
        factors.factor.between(*config["forecast"]["factor_bounds"]).all(),
    )
    forecast_scores = pd.read_csv(output / "forecast/forecast_metrics_extended.csv")
    choices = pd.read_csv(output / "forecast/forecast_choices_extended.csv")
    for j in range(1, 5):
        scores = forecast_scores.loc[
            forecast_scores["product"].eq(j)
            & forecast_scores["sample"].eq("retrospective_validation")
            & forecast_scores.promotion.eq(-1)
        ].set_index("model")
        expected = min(
            config["forecast"]["models"], key=lambda model: scores.loc[model, "WAPE"]
        )
        check(
            f"product {j}: validation-only model selection",
            choices.loc[choices["product"].eq(j), "model"].iloc[0] == expected,
        )
        predictions = pd.read_csv(
            output / f"forecast/forecast_predictions_product{j}.csv"
        )
        Ns = (
            forecast_scores.loc[
                forecast_scores["product"].eq(j) & forecast_scores.promotion.eq(-1)
            ]
            .groupby("sample")
            .N.nunique()
        )
        check(f"product {j}: common forecast scoring rows", Ns.eq(1).all())
        check(
            f"product {j}: nonnegative forecasts",
            predictions[config["forecast"]["models"]].min().min() >= 0,
        )
        origin = 350
        calendar, features = features_for(j)
        fitting = (
            panel.week.lt(origin).to_numpy()
            & panel.week.ge(origin - config["forecast"]["rolling_window"]).to_numpy()
        )
        target = panel.week.eq(origin).to_numpy()
        values, fit, smear = forecast_levels(panel, f"zq{j}", features, fitting, target)
        close(
            f"product {j}: refit rolling prediction",
            values,
            predictions.loc[target, "rolling_no_trend"],
        )
        close(
            f"product {j}: fitting-window smearing",
            smear,
            np.exp(fit.residual[:, 0]).mean(),
            1e-12,
        )
        altered = panel.copy()
        # Rebuild lags after contaminating all unavailable future outcomes.
        for item in range(1, 5):
            altered.loc[altered.week.ge(origin), f"move{item}"] = 1e8
            altered[f"zq{item}"] = np.log1p(altered[f"move{item}"])
            altered[f"lag1_{item}"] = calendar_lag(altered, f"zq{item}", 1)
            altered[f"lag52z_{item}"] = calendar_lag(altered, f"zq{item}", 52)
        changed, _, _ = forecast_levels(altered, f"zq{j}", features, fitting, target)
        close(f"product {j}: future-outcome perturbation", changed, values)
        base = predictions.rolling_no_trend.to_numpy()
        original, _ = calibrate_origins(panel, j, base, config["forecast"], True)
        changed, _ = calibrate_origins(altered, j, base, config["forecast"], True)
        close(
            f"product {j}: calibration future-outcome perturbation",
            original[target],
            changed[target],
        )
    pricing = pd.read_csv(output / "pricing/pricing_sensitivity_store.csv")
    details = pd.read_csv(output / "pricing/pricing_sensitivity_details.csv")
    keys = [
        "system",
        "specification",
        "price_rule",
        "maxchange",
        "cost_multiplier",
        "store",
        "week",
        "policy",
    ]
    details["calculated_profit"] = (details.price - details.cost) * details.quantity
    summed = details.groupby(keys).calculated_profit.sum()
    close(
        "portfolio profit includes all product/cross-price contributions",
        pricing.set_index(keys).profit.sort_index(),
        summed.sort_index(),
    )
    check(
        "prices: box support",
        (
            details.price.ge(details.lower_bound - 1e-8)
            & details.price.le(details.upper_bound + 1e-8)
        ).all(),
    )
    policy_values = pricing.pivot(index=keys[:-1], columns="policy", values="profit")
    check(
        "joint profit >= observed profit",
        (
            policy_values.joint >= policy_values.observed - settings["profit_tolerance"]
        ).all(),
    )
    check(
        "joint profit >= feasible independent benchmark",
        (
            policy_values.joint
            >= policy_values.independent - settings["profit_tolerance"]
        ).all(),
    )
    prices_wide = details.pivot(index=keys, columns="product", values="price")
    baselines = details.pivot(index=keys, columns="product", values="baseline_price")
    linked = prices_wide.index.get_level_values("price_rule") == "pair_linked"
    ratios = prices_wide[linked] / baselines[linked]
    close("linked Pepsi proportional changes", ratios[1], ratios[2], 1e-8)
    close("linked Coke proportional changes", ratios[3], ratios[4], 1e-8)
    limits = pd.read_csv(
        output / "pricing/relative_price_constraints.csv"
    ).drop_duplicates(["store", "maxchange", "product_a", "product_b"])
    for _, limit in limits.iterrows():
        target = (prices_wide.index.get_level_values("store") == limit.store) & (
            prices_wide.index.get_level_values("maxchange") == limit.maxchange
        )
        relative = np.log(
            prices_wide.loc[target, int(limit.product_a)]
            / prices_wide.loc[target, int(limit.product_b)]
        )
        check(
            f"relative price support: store {int(limit.store)} limit {limit.maxchange} pair {int(limit.product_a)}-{int(limit.product_b)}",
            relative.between(
                limit.lower_log_ratio - 1e-8, limit.upper_log_ratio + 1e-8
            ).all(),
        )
    search = pd.read_csv(output / "pricing/search_audit.csv")
    for _, row in search.iterrows():
        check(
            f"search global grid comparison: {row.system} {row.specification}",
            row.local_global_gap <= 1e-6,
            f"gap {row.local_global_gap:.8g}; one store audited",
            warning=True,
        )
    reference = pd.read_csv(output / "pricing/pricing_reference_stores.csv")
    ref = panel.loc[
        panel.week.eq(reference.week.iloc[0]) & panel.store.isin(reference.store)
    ]
    index = ref.index[0]
    row = ref.iloc[0]
    for key, model in models.items():
        B, q0 = model_inputs(model, index, row, weights)
        p0 = row[[f"p{j}" for j in range(1, 5)]].to_numpy(float)
        cost = row[[f"cost{j}" for j in range(1, 5)]].to_numpy(float)
        analytic = q0 + (B.T @ ((p0 - cost) * q0)) / p0
        numeric = []
        for j in range(4):
            step = 1e-5 * p0[j]
            plus = p0.copy()
            minus = p0.copy()
            plus[j] += step
            minus[j] -= step
            numeric.append(
                (profit(plus, p0, q0, cost, B)[0] - profit(minus, p0, q0, cost, B)[0])
                / (2 * step)
            )
        close(f"profit derivatives with cross terms: {key}", numeric, analytic, 1e-5)
    robust = pd.read_csv(output / "pricing/robust_price_details.csv")
    for store, frame in robust.groupby("store"):
        row = ref.loc[ref.store.eq(store)].iloc[0]
        p0, lo, hi, limits, _ = support_for_store(
            panel.loc[panel.week.le(row.week)],
            row,
            config["pricing"]["robust_maximum_change"],
            config["pricing"]["support_quantiles"],
        )
        prices = frame.sort_values("product").price.to_numpy()
        check(
            f"robust policy feasible: store {int(store)}",
            feasible(prices, lo, hi, limits)[0],
        )
    robust_summary = pd.read_csv(output / "pricing/robust_policy_aggregate.csv")
    check(
        "robust screen covers every estimated demand specification",
        len(robust_summary) == len(models) * len(config["pricing"]["cost_multipliers"]),
    )
    check("robust aggregate lower-tail gains", robust_summary.gain_p05.ge(-1e-6).all())
    comparison = []
    previous = config["paths"]["reference"] / "previous_elasticities.csv"
    if previous.exists():
        old = pd.read_csv(previous)
        new = pd.read_csv(output / "demand/elasticities.csv")
        joined = old.merge(
            new,
            on=["demand_product", "price_product", "specification"],
            suffixes=("_old", "_new"),
        )
        for _, row in joined.iterrows():
            comparison.append(
                dict(
                    component="demand point estimate",
                    item=f"{row.specification} {int(row.demand_product)}<-{int(row.price_product)}",
                    previous=row.elasticity_old,
                    revised=row.elasticity_new,
                    change=row.elasticity_new - row.elasticity_old,
                    explanation="Source float64 replaces legacy float32; inference now clusters store and week",
                )
            )
        close(
            "unchanged unrestricted point estimates vs prior",
            joined.elasticity_new,
            joined.elasticity_old,
        )
        comparison.append(
            dict(
                component="inference",
                item="Pepsi own-price SE, week FE",
                previous=joined.iloc[0].se_old,
                revised=joined.iloc[0].se_new,
                change=joined.iloc[0].se_new - joined.iloc[0].se_old,
                explanation="Intentional store-and-week clustering; no exact legacy-SE requirement",
            )
        )
    comparison.extend(
        [
            dict(
                component="sample",
                item="balanced store-weeks",
                previous=31378,
                revised=len(panel),
                change=len(panel) - 31378,
                explanation="Same selected UPCs and training-history rule",
            ),
            dict(
                component="market coverage",
                item="training revenue share (%)",
                previous=71.12,
                revised=100
                * coverage.loc[coverage.split.eq(1), "market_coverage"].iloc[0],
                change=100
                * coverage.loc[coverage.split.eq(1), "market_coverage"].iloc[0]
                - 71.12,
                explanation="Prior report's rounded Coke/Pepsi-only share vs expanded 2L cola denominator; intentional change",
            ),
            dict(
                component="dynamic sample",
                item="lag-complete store-weeks",
                previous=np.nan,
                revised=models[("linked_product", "distributed_lag")].fit.n,
                change=np.nan,
                explanation="New four-week distributed-lag specification; aligned static comparison uses the same rows",
            ),
        ]
    )
    previous_forecasts = config["paths"]["reference"] / "previous_forecast_metrics.csv"
    if previous_forecasts.exists():
        old = pd.read_csv(previous_forecasts)
        old = old.loc[
            old["sample"].eq("retrospective_test")
            & old.promotion.eq(-1)
            & old.model.eq("rolling_no_trend")
        ]
        new = forecast_scores.loc[
            forecast_scores["sample"].eq("retrospective_test")
            & forecast_scores.promotion.eq(-1)
            & forecast_scores.model.eq("rolling_no_trend")
        ]
        joined = old.merge(new, on=["product", "model"], suffixes=("_old", "_new"))
        for _, row in joined.iterrows():
            for statistic in ["WAPE", "bias_pct"]:
                comparison.append(
                    dict(
                        component="forecast baseline",
                        item=f"product {int(row['product'])} {statistic}",
                        previous=row[statistic + "_old"],
                        revised=row[statistic + "_new"],
                        change=row[statistic + "_new"] - row[statistic + "_old"],
                        explanation="Common main scoring sample preserved; float64 inputs replace legacy float32",
                    )
                )
    previous_pricing = config["paths"]["reference"] / "previous_pricing_aggregate.csv"
    if previous_pricing.exists():
        old = pd.read_csv(previous_pricing)
        old = old.loc[
            old.system.eq("product")
            & old.specification.eq("week_FE")
            & old.price_rule.eq("pair_linked")
            & old.maxchange.eq(0.15)
            & old.cost_multiplier.eq(1)
            & old.policy.eq("joint")
        ]
        new = pd.read_csv(output / "pricing/pricing_sensitivity_aggregate.csv")
        new = new.loc[
            new.system.eq("linked_product")
            & new.specification.eq("week_FE")
            & new.maxchange.eq(0.15)
            & new.cost_multiplier.eq(1)
            & new.policy.eq("joint")
        ]
        comparison.append(
            dict(
                component="main pricing",
                item="joint gross-contribution gain (%)",
                previous=old.gain_pct.iloc[0],
                revised=new.gain_pct.iloc[0],
                change=new.gain_pct.iloc[0] - old.gain_pct.iloc[0],
                explanation="Unrestricted demand with linked decisions replaced by directly linked demand and relative support; intentional estimand change",
            )
        )
    save_csv(comparison, output, "methodological_discrepancies.csv")
    checks = save_csv(rows, output, "verification_checks.csv")
    summary = dict(
        checks=len(checks),
        passed=int(checks.status.eq("PASS").sum()),
        failed=int(checks.status.eq("FAIL").sum()),
        warnings=int(checks.status.eq("WARN").sum()),
        absolute_tolerance=settings["absolute_tolerance"],
        relative_tolerance=settings["relative_tolerance"],
        limitations=[
            "Windows was not executed",
            "No fresh untouched forecast period remains",
            "No experimental assignment file",
            "Uncertainty conditional on estimated baseline levels and selected market",
            "Unrestricted global search audited at selected examples only; no continuous global optimum certified",
        ],
    )
    save_json(summary, output / "verification_report.json")
    return summary
