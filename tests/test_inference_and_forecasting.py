"""Independent inference and operational information-set benchmarks."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from statsmodels.sandbox.regression.gmm import IV2SLS
from retail.equation_inference import absorb, fit_iv, fit_ols
from retail.econometrics import group_sum
from retail.forecast_operational import (
    baseline_forecast,
    construct_forecast_data,
    feature_names,
    predict_forecast,
)


def synthetic_panel(seed=1729):
    rng = np.random.default_rng(seed)
    store = np.repeat(np.arange(24), 40)
    week = np.tile(np.arange(40), 24)
    keep = rng.random(len(store)) > 0.08
    store, week = store[keep], week[keep]
    x = rng.normal(size=(len(store), 3))
    residual = (
        0.4 * rng.normal(size=len(store))
        + 0.2 * rng.normal(size=24)[store]
        + 0.3 * rng.normal(size=40)[week]
    )
    y = (
        x @ np.array([-1.4, 0.3, 0.1])
        + rng.normal(size=24)[store]
        + rng.normal(size=40)[week]
        + residual
    )
    return y, x, store, week


def test_absorbed_ols_and_two_way_inference_against_explicit_dummies():
    y, x, store, week = synthetic_panel()
    fit = fit_ols(y, x, ["price_p1", "price_p2", "promotion_p1"], store, week)
    fixed = pd.get_dummies(
        pd.DataFrame({"store": store.astype(str), "week": week.astype(str)}),
        drop_first=True,
        dtype=float,
    )
    design = sm.add_constant(np.column_stack([x, fixed]))
    explicit = sm.OLS(y, design).fit()
    covariance, _, _ = cov_cluster_2groups(explicit, store, week)
    assert np.allclose(fit["coefficients"], explicit.params[1:4], atol=1e-8, rtol=1e-6)
    assert np.allclose(fit["covariance"], covariance[1:4, 1:4], atol=1e-8, rtol=1e-6)
    transformed, _, _ = absorb(np.column_stack([y, x]), [store, week])
    for group in [store, week]:
        codes = np.unique(group, return_inverse=True)[1]
        means = group_sum(transformed, codes) / np.bincount(codes)[:, None]
        assert np.max(abs(means)) < 1e-8


def test_iv_against_independent_two_stage_least_squares():
    rng = np.random.default_rng(47)
    store = np.repeat(np.arange(24), 40)
    week = np.tile(np.arange(40), 24)
    z = rng.normal(size=(len(store), 4))
    controls = rng.normal(size=(len(store), 4))
    disturbance = rng.normal(size=len(store))
    price = z + 0.4 * controls + 0.4 * disturbance[:, None]
    beta = np.array([-1.3, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4])
    x = np.column_stack([price, controls])
    y = x @ beta + disturbance + rng.normal(size=24)[store] + rng.normal(size=40)[week]
    names = [f"price_p{i}" for i in range(1, 5)] + [
        f"promotion_p{i}" for i in range(1, 5)
    ]
    fit, _ = fit_iv(y, price, controls, z, names, store, week)
    within, _, _ = absorb(np.column_stack([y, x, z]), [store, week])
    instruments = np.column_stack([within[:, 9:13], within[:, 5:9]])
    reference = IV2SLS(within[:, 0], within[:, 1:9], instruments).fit()
    assert np.allclose(fit["coefficients"], reference.params, atol=1e-8, rtol=1e-6)


def test_wild_score_update_matches_refitting_with_fixed_effects():
    y, x, store, week = synthetic_panel()
    fit = fit_ols(y, x, ["price_p1", "price_p2", "promotion_p1"], store, week)
    codes = np.unique(store, return_inverse=True)[1]
    multiplier = np.random.default_rng(3).choice([-1, 1], 24)
    scores = group_sum(fit["x_within"] * fit["residuals"][:, None], codes)
    score_beta = fit["coefficients"] + multiplier @ scores @ fit["bread"]
    bootstrap_y = fit["fitted"] + fit["residuals"] * multiplier[codes]
    transformed, _, _ = absorb(np.column_stack([bootstrap_y, x]), [store, week])
    refit_beta = np.linalg.lstsq(transformed[:, 1:], transformed[:, 0], rcond=None)[0]
    assert np.allclose(score_beta, refit_beta, atol=1e-8, rtol=1e-6)


def forecast_panel():
    data = pd.DataFrame(
        {"store": np.repeat([1, 2], 80), "week": np.tile(np.arange(1, 81), 2)}
    )
    data["income"] = 10.5
    data["customers"] = 1000
    for i in range(1, 5):
        data[f"units_p{i}"] = 20 + i + data.week * 0.1
        data[f"price_p{i}"] = 1 + i * 0.1
        data[f"promotion_p{i}"] = 0
    return data


def test_operational_features_respect_calendar_and_future_mutation():
    raw = forecast_panel()
    before = construct_forecast_data(raw)
    changed = raw.copy()
    mask = changed.week.ge(65)
    changed.loc[mask, "units_p1"] *= 10
    changed.loc[mask, "price_p1"] *= 10
    after = construct_forecast_data(changed)
    columns, _ = feature_names("operational")
    pd.testing.assert_frame_equal(
        before[before.week.le(65)][columns].reset_index(drop=True),
        after[after.week.le(65)][columns].reset_index(drop=True),
    )
    missing = construct_forecast_data(raw[~((raw.store == 1) & (raw.week == 62))])
    assert not ((missing.store == 1) & (missing.week == 63)).any()


def test_log1p_smearing_subtracts_shift_after_retransformation():
    class ConstantPrediction:
        def predict(self, data):
            return np.repeat(np.log(11), len(data))

    data = pd.DataFrame({"product": ["p1", "p2"]})
    model = dict(
        model=ConstantPrediction(),
        columns=["product"],
        logarithmic=True,
        smearing={"p1": 1.2, "p2": 1.1},
    )
    assert np.allclose(predict_forecast(model, data), [12.2, 11.1])


def test_unavailable_annual_lag_uses_only_observed_recent_sales():
    raw = forecast_panel()
    data = construct_forecast_data(raw[~((raw.store == 1) & (raw.week == 12))])
    target = data[(data.store == 1) & (data.week == 64)]
    assert len(target) == 4 and target.lag52.isna().all()
    assert np.allclose(baseline_forecast(target, "seasonal_52_weeks"), target.mean4)


def test_iv_two_way_covariance_matches_full_dummy_instrument_calculation():
    rng = np.random.default_rng(47)
    store = np.repeat(np.arange(24), 40)
    week = np.tile(np.arange(40), 24)
    z = rng.normal(size=(len(store), 4))
    controls = rng.normal(size=(len(store), 4))
    disturbance = rng.normal(size=len(store))
    price = z + 0.4 * controls + 0.4 * disturbance[:, None]
    x = np.column_stack([price, controls])
    beta = np.array([-1.3, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4])
    y = x @ beta + disturbance + rng.normal(size=24)[store] + rng.normal(size=40)[week]
    names = [f"price_p{i}" for i in range(1, 5)] + [
        f"promotion_p{i}" for i in range(1, 5)
    ]
    fit, _ = fit_iv(y, price, controls, z, names, store, week)
    dummies = sm.add_constant(
        pd.get_dummies(
            pd.DataFrame({"store": store.astype(str), "week": week.astype(str)}),
            drop_first=True,
            dtype=float,
        ).to_numpy()
    )
    design = np.column_stack([x, dummies])
    instruments = np.column_stack([z, controls, dummies])
    projected = instruments @ np.linalg.lstsq(instruments, design, rcond=None)[0]
    bread = np.linalg.inv(projected.T @ design)
    estimate = bread @ projected.T @ y
    residual = y - design @ estimate
    scores = projected * residual[:, None]
    n, k = design.shape

    def covariance(group):
        codes = np.unique(group, return_inverse=True)[1]
        sums = np.column_stack(
            [np.bincount(codes, weights=scores[:, j]) for j in range(k)]
        )
        clusters = len(sums)
        return (
            bread
            @ (sums.T @ sums)
            @ bread.T
            * clusters
            / (clusters - 1)
            * (n - 1)
            / (n - k)
        )

    intersections = np.unique(
        np.column_stack([store, week]), axis=0, return_inverse=True
    )[1]
    full = covariance(store) + covariance(week) - covariance(intersections)
    np.testing.assert_allclose(fit["covariance"], full[:8, :8], atol=1e-8, rtol=1e-6)


def test_operational_preprocessing_and_smearing_use_fitting_rows_only():
    from retail.forecast_operational import fit_forecast_model

    raw = forecast_panel()
    data = construct_forecast_data(raw)
    data.loc[data.week.eq(55), "log_traffic_lag1"] = np.nan
    training = data[data.week.le(64)]
    config = {
        "forecast": {
            "threads": 1,
            "random_seed": 47,
            "forest_trees": 10,
            "boosting_iterations": 10,
        }
    }
    fit = fit_forecast_model(training, "ridge", 10.0, "operational", config)
    columns, numeric = feature_names("operational")
    preprocessing = fit["model"].named_steps["preprocessing"]
    transform = preprocessing.named_transformers_["numeric"]
    expected = training[numeric].median().to_numpy()
    np.testing.assert_allclose(
        transform.named_steps["impute"].statistics_, expected, atol=1e-12, rtol=1e-12
    )
    imputed = training[numeric].fillna(training[numeric].median()).to_numpy()
    np.testing.assert_allclose(
        transform.named_steps["scale"].mean_,
        imputed.mean(axis=0),
        atol=1e-12,
        rtol=1e-12,
    )
    residual = np.log1p(training.target) - fit["model"].predict(training[columns])
    factors = (
        pd.Series(np.exp(residual.to_numpy()))
        .groupby(training["product"].to_numpy())
        .mean()
        .to_dict()
    )
    for product in factors:
        assert np.isclose(
            fit["smearing"][product], factors[product], atol=1e-12, rtol=1e-12
        )
    assert fit["train_through"] == 64
