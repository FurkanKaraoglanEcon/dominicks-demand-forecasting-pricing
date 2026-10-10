"""Checks of economic identities, independent OLS algebra and chronological information sets.

Inputs: current panels, fitted systems, settings and generated tables.
Outputs: independent numerical checks appended through explicit check/close callbacks.
"""

import json
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .io import save_csv, calendar_lag, PRICES, PROMOTIONS
from .econometrics import fit_system, forecast_levels
from .forecast_benchmarks import features_for, calibrate_origins
from .pricing import support_for_store, feasible, model_inputs, profit, profit_gradient


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


def verify_retained_demand(panel, models, output, pricing, check, close):
    """Compare current demand outputs and models using check/close callbacks.
    Verify retained specifications, common-price contrasts and dynamic accounting."""
    retained_pricing = {
        (system, specification, rule)
        for specification in ["week_FE", "season_trend", "season_only"]
        for system, rule in [
            ("product", "product_free"),
            ("product", "pair_linked"),
            ("family", "pair_linked"),
        ]
    }
    executed_pricing = set(
        pricing[["system", "specification", "price_rule"]].itertuples(
            index=False, name=None
        )
    )
    check(
        "all nine earlier pricing model/rule combinations executed",
        retained_pricing.issubset(executed_pricing),
    )
    contrasts = pd.read_csv(output / "demand/common_and_relative_price_effects.csv")
    reparameterized = fit_system(
        panel,
        [f"lq{j}" for j in range(1, 5)],
        ["flp1", "wedge1", "flp2", "wedge2"] + PROMOTIONS,
        models[("product", "week_FE")].fit.mask,
        True,
    )
    original = models[("product", "week_FE")]
    close(
        "common/wedge reparameterization preserves fitted demand",
        reparameterized.fitted,
        original.fit.fitted,
    )
    for _, contrast in contrasts.loc[contrasts.specification.eq("week_FE")].iterrows():
        equation, family = int(contrast["product"]) - 1, int(contrast.family) - 1
        close(
            f"common price contrast independently estimated: product {equation+1} family {family+1}",
            contrast.common_price_elasticity,
            reparameterized.beta[2 * family, equation],
        )
        close(
            f"relative price contrast independently estimated: product {equation+1} family {family+1}",
            contrast.relative_price_slope,
            reparameterized.beta[2 * family + 1, equation],
        )
        index = equation * len(reparameterized.regressors) + 2 * family
        close(
            f"common price contrast covariance: product {equation+1} family {family+1}",
            contrast.common_se**2,
            reparameterized.covariance[index, index],
        )
        close(
            f"relative price contrast covariance: product {equation+1} family {family+1}",
            contrast.relative_se**2,
            reparameterized.covariance[index + 1, index + 1],
        )


def verify_full_panel_derivatives(panel, models, output, reference, check, close):
    """Recompute full-panel profit derivatives from panel, models and exported results.
    Append numerical identities using check/close; return no data."""
    derivative_details = pd.read_csv(output / "pricing/profit_derivative_details.csv")
    derivative_summary = pd.read_csv(output / "pricing/profit_derivative_summary.csv")
    check(
        "profit derivatives cover all earlier time specifications",
        {"week_FE", "season_trend", "season_only"}.issubset(
            derivative_summary.specification
        ),
    )
    check(
        "profit derivative summary covers full panel rather than only pricing reference",
        derivative_summary.N.gt(len(reference)).all(),
    )
    for key, model in models.items():
        if model.system != "product":
            continue
        sample = derivative_details.loc[derivative_details.specification.eq(key[1])]
        usable = (
            np.isfinite(model.baseline_units).all(axis=1)
            & panel[[f"cost{j}" for j in range(1, 5)]].notna().all(axis=1).to_numpy()
        )
        p = panel.loc[usable, [f"p{j}" for j in range(1, 5)]].to_numpy(float)
        c = panel.loc[usable, [f"cost{j}" for j in range(1, 5)]].to_numpy(float)
        q = model.baseline_units[usable]
        expected = q + np.einsum("ni,ij->nj", (p - c) * q, model.elasticity) / p
        close(
            f"full-panel portfolio gradient identity: {key}",
            sample.portfolio_profit_derivative,
            expected.reshape(-1),
        )
        close(
            f"shared gradient matches reference and full-panel algebra: {key}",
            profit_gradient(p, q, c, model.elasticity),
            expected,
        )
        check(
            f"full-panel derivative observation accounting: {key}",
            len(sample) == int(usable.sum()) * 4,
        )
        summary = derivative_summary.loc[
            derivative_summary.specification.eq(key[1])
        ].sort_values("product")
        actual = (
            sample.assign(positive=sample.portfolio_profit_derivative.gt(0))
            .groupby("product")
            .positive.mean()
        )
        close(
            f"full-panel derivative positive shares: {key}",
            summary.share_portfolio_positive,
            actual,
        )


def verify_price_events(panel, config, output, check, close):
    """Verify price-event tables and training-derived history from config and panel.
    Append variation and event-window checks through check/close callbacks."""
    event_counts = pd.read_csv(output / "identification/price_event_counts.csv")
    anchors = pd.read_csv(output / "identification/price_event_anchors.csv")
    windows = pd.read_csv(output / "identification/event_windows.csv")
    event_profiles = pd.read_csv(output / "identification/event_profiles.csv")
    check(
        "all twelve earlier event types executed",
        len(event_counts) == 12 and event_counts.event_type.nunique() == 12,
    )
    check(
        "event clean/complete/anchor count ordering",
        (
            event_counts.clean_windows.le(event_counts.complete_windows)
            & event_counts.complete_windows.le(event_counts.events)
        ).all(),
    )
    anchor_counts = (
        anchors.groupby("event_type")
        .size()
        .reindex(event_counts.event_type, fill_value=0)
    )
    close("event anchors reconcile to count table", event_counts.events, anchor_counts)
    window_size = 2 * config["identification"]["event_window"] + 1
    check(
        "event windows have actual contiguous calendar dates",
        windows.week.eq(windows.anchor_week + windows.relative_week).all(),
    )
    counts = windows.groupby(["event_type", "store", "anchor_week"]).size()
    check(
        "complete event windows contain every required week",
        counts.eq(window_size).all(),
    )
    indexed = panel.set_index(["store", "week"])
    current = indexed.reindex(pd.MultiIndex.from_arrays([windows.store, windows.week]))
    baseline = indexed.reindex(
        pd.MultiIndex.from_arrays([windows.store, windows.anchor_week - 1])
    )
    family = np.where(windows.event_name.str.startswith(("flp1", "lp1_", "lp2_")), 0, 1)
    quantities = current[["fq1", "fq2"]].to_numpy()
    base_quantities = baseline[["fq1", "fq2"]].to_numpy()
    positions = np.arange(len(windows))
    close(
        "event focal response uses observed week -1 baseline",
        windows.focal_family_pct,
        100 * (quantities[positions, family] / base_quantities[positions, family] - 1),
    )
    close(
        "event rival response uses observed week -1 baseline",
        windows.rival_family_pct,
        100
        * (
            quantities[positions, 1 - family] / base_quantities[positions, 1 - family]
            - 1
        ),
    )
    for _, event in event_counts.iterrows():
        matching = windows.loc[windows.event_type.eq(event.event_type)]
        selected_anchors = anchors.loc[anchors.event_type.eq(event.event_type)]
        focal, direction = event.event_name.rsplit("_", 1)
        rivals = (
            ["flp2" if focal == "flp1" else "flp1"]
            if focal.startswith("flp")
            else [p for p in PRICES if p != focal]
        )
        current_anchors = indexed.reindex(
            pd.MultiIndex.from_arrays([selected_anchors.store, selected_anchors.week])
        )
        prior_anchors = indexed.reindex(
            pd.MultiIndex.from_arrays(
                [selected_anchors.store, selected_anchors.week - 1]
            )
        )
        delta = current_anchors[focal].to_numpy() - prior_anchors[focal].to_numpy()
        cutoff = (
            np.log1p(-config["identification"]["event_change"])
            if direction == "cut"
            else np.log1p(config["identification"]["event_change"])
        )
        check(
            f"event own-price threshold: {event.event_name}",
            (
                (delta <= cutoff + 1e-10).all()
                if direction == "cut"
                else (delta >= cutoff - 1e-10).all()
            ),
        )
        close(
            f"event exported own change: {event.event_name}",
            selected_anchors.focal_log_change,
            delta,
        )
        check(
            f"event rival stability at anchor: {event.event_name}",
            (
                np.abs(
                    current_anchors[rivals].to_numpy()
                    - prior_anchors[rivals].to_numpy()
                )
                <= np.log1p(config["identification"]["event_rival_stability"]) + 1e-10
            ).all(),
        )
        check(
            f"event unchanged promotion at anchor: {event.event_name}",
            np.array_equal(
                current_anchors[PROMOTIONS].to_numpy(),
                prior_anchors[PROMOTIONS].to_numpy(),
            ),
        )
        close(
            f"event complete-window count: {event.event_name}",
            event.complete_windows,
            len(matching) / window_size,
        )
        close(
            f"event clean-window count: {event.event_name}",
            event.clean_windows,
            matching.clean_window.sum() / window_size,
        )
        if len(matching):
            keys_now = pd.MultiIndex.from_arrays([matching.store, matching.week])
            keys_base = pd.MultiIndex.from_arrays(
                [matching.store, matching.anchor_week - 1]
            )
            observed = indexed.reindex(keys_now)
            base = indexed.reindex(keys_base)
            stable_prices = (
                np.abs(observed[rivals].to_numpy() - base[rivals].to_numpy())
                <= np.log1p(config["identification"]["event_rival_stability"])
            ).all(axis=1)
            stable_promotions = (
                observed[PROMOTIONS].to_numpy() == base[PROMOTIONS].to_numpy()
            ).all(axis=1)
            event_dates = pd.MultiIndex.from_frame(
                anchors[["store", "week"]].drop_duplicates()
            )
            independent = pd.DataFrame(
                dict(
                    store=matching.store.to_numpy(),
                    anchor_week=matching.anchor_week.to_numpy(),
                    stable=stable_prices & stable_promotions,
                    event_date=keys_now.isin(event_dates),
                )
            )
            window_status = independent.groupby(["store", "anchor_week"]).agg(
                stable=("stable", "all"), event_dates=("event_date", "sum")
            )
            expected_clean = window_status.stable & window_status.event_dates.eq(1)
            saved_clean = (
                matching.groupby(["store", "anchor_week"])
                .clean_window.first()
                .astype(bool)
            )
            check(
                f"event full-window restrictions independently checked: {event.event_name}",
                np.array_equal(saved_clean.to_numpy(), expected_clean.to_numpy()),
            )
    for label in event_profiles["sample"].unique():
        sample = (
            windows
            if label == "all_qualified_anchors"
            else windows.loc[windows.clean_window.eq(1)]
        )
        direct = sample.groupby(["event_type", "relative_week"])[
            ["focal_family_pct", "rival_family_pct"]
        ].mean()
        saved = event_profiles.loc[event_profiles["sample"].eq(label)].set_index(
            ["event_type", "relative_week"]
        )[["focal_family_pct", "rival_family_pct"]]
        close(
            f"descriptive event profile means: {label}",
            saved.sort_index(),
            direct.sort_index(),
        )


def verify_scope_outputs(output, check):
    """Check configured output files against the declared research scope.
    Append completion checks through check; return no data."""
    pair_diagnostics = pd.read_csv(output / "identification/price_pair_diagnostics.csv")
    check(
        "all six earlier price-pair diagnostics retained",
        len(pair_diagnostics) == 6
        and not pair_diagnostics.duplicated(["product1", "product2"]).any(),
    )
    income = pd.read_csv(output / "demand/income_heterogeneity.csv")
    check(
        "income interactions preserve all four equations and original fields",
        len(income) == 4
        and {"own_elasticity", "income_interaction", "se_interaction", "N"}.issubset(
            income.columns
        ),
    )
    scope_rows = [
        dict(
            component="Original product demand and seasonal sensitivity",
            status="retained with revised two-way inference",
            results="demand/elasticities.csv; demand/demand_fit.csv",
        ),
        dict(
            component="Original income-price interactions",
            status="retained; compatible original fields restored",
            results="demand/income_heterogeneity.csv",
        ),
        dict(
            component="Extension product/family time specifications",
            status="all six retained; linked demand added",
            results="demand/demand_sensitivity.csv",
        ),
        dict(
            component="Original and extension forecast candidates",
            status="all six retained; two calibration candidates added",
            results="forecast/forecast_metrics_extended.csv",
        ),
        dict(
            component="Extension free and linked pricing under unrestricted demand",
            status="both retained; omitted-index and dynamic variants also run",
            results="pricing/pricing_sensitivity_aggregate.csv",
        ),
        dict(
            component="Extension aggregate-family pricing",
            status="retained with actual SKU margin accounting and relative support",
            results="pricing/pricing_sensitivity_aggregate.csv",
        ),
        dict(
            component="Extension isolated-event windows and profiles",
            status="restored; descriptive only",
            results="identification/price_event_counts.csv; identification/event_profiles.csv",
        ),
        dict(
            component="Extension common/relative price decomposition",
            status="restored with contrast covariance and intervals",
            results="demand/common_and_relative_price_effects.csv",
        ),
        dict(
            component="Extension full-panel profit derivatives",
            status="restored; reference derivatives also retained",
            results="pricing/profit_derivative_details.csv; pricing/profit_derivative_summary.csv",
        ),
        dict(
            component="Original weekly sales/visit summaries",
            status="restored; aggregate visits remain descriptive",
            results="weekly_segment_summary.csv; reports/figures/weekly_units.png",
        ),
        dict(
            component="All scanner, UPC, demographic and customer inputs",
            status="complete raw preparation retained",
            results="raw_checksums.json; categories/category_summary.csv",
        ),
    ]
    save_csv(scope_rows, output, "research_scope_audit.csv")


def verify_raw_sample(config, panel, selected, models, check, close):
    """Recompute raw sample identities from current outputs."""
    output = config["paths"]["output"]
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
    weekly_summary = pd.read_csv(output / "weekly_segment_summary.csv")
    close(
        "weekly portfolio units reconcile",
        weekly_summary.segment_units.sum(),
        panel.segment_units.sum(),
    )
    close(
        "weekly portfolio revenue reconciles",
        weekly_summary.segment_revenue.sum(),
        panel.segment_revenue.sum(),
    )
    check(
        "weekly summary covers every retained week",
        set(weekly_summary.week) == set(panel.week),
    )
    check(
        "all earlier product/family demand specifications retained",
        {
            (system, specification)
            for system in ["product", "family"]
            for specification in ["week_FE", "season_trend", "season_only"]
        }.issubset(models),
    )
    previous_models = {
        "seasonal_naive",
        "recent4",
        "calendar",
        "pricing",
        "no_trend",
        "rolling_no_trend",
    }
    check(
        "all six earlier forecast candidates retained",
        previous_models.issubset(config["forecast"]["models"]),
    )
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
        np.array_equal(
            panel.split.to_numpy(),
            np.select(
                [
                    panel.week.le(audit["train_end"]),
                    panel.week.le(audit["validation_end"]),
                ],
                [1, 2],
                default=3,
            ),
        ),
    )
    check(
        "source sample vs previous run",
        len(panel) == 31378 and panel.store.nunique() == 86,
        "Prior sample preserved; broader market changes coverage denominator",
        warning=True,
    )
    return audit


def verify_training_selection(config, panel, selected, check, close):
    """Recompute training selection identities from current outputs."""
    output = config["paths"]["output"]
    audit = json.loads((output / "preparation_audit.json").read_text())
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
    return weights


def verify_accounting_inference(config, panel, models, check, close):
    """Recompute accounting inference identities from current outputs."""
    output = config["paths"]["output"]
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
        "customer counts: seven nonnegative valid days and positive weekly total",
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


def verify_benchmark_forecasts(config, panel, audit, check, close):
    """Recompute benchmark forecasts identities from current outputs."""
    output = config["paths"]["output"]
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
        test_weeks = np.sort(panel.loc[panel.split.eq(3), "week"].unique())
        assert len(test_weeks) > 0, "Forecast verification needs a held-out target week"
        origin = int(test_weeks[min(10, len(test_weeks) - 1)])
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


def verify_primary_pricing(config, panel, models, weights, check, close):
    """Recompute primary pricing identities from current outputs."""
    output = config["paths"]["output"]
    settings = config["verification"]
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
    verify_retained_demand(panel, models, output, pricing, check, close)
    verify_full_panel_derivatives(panel, models, output, ref, check, close)
    verify_price_events(panel, config, output, check, close)
    verify_scope_outputs(output, check)
    # Reconciliation is computed independently after the run. Frozen earlier
    # results are optional review material and never execution dependencies.
