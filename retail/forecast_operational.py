"""Chronological, one-week-ahead forecasts with pretest model selection.

Inputs: product/store/week quantities, prices, recorded promotions and traffic.
Outputs: validation predictions, rolling test forecasts, calibration/accuracy
tables, information-set audits and fitted models. Operational forecasts use
information dated t-1 or earlier; the separately labelled conditional exercise
also uses realized t prices/promotions and is a retrospective policy forecast.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io import stage_log, table_path
from .product_definitions import PRODUCTS, PRICE_NAMES, PROMOTION_NAMES

NUMERIC_FEATURES = [
    "log_lag1",
    "log_mean4",
    "log_lag52",
    "log_other_lag1",
    "log_traffic_lag1",
    "income",
    "season_sin",
    "season_cos",
    "trend",
]
LAG_PRICE_FEATURES = [f"log_price_lag1_{p}" for p in PRODUCTS]
LAG_PROMOTION_FEATURES = [f"promotion_lag1_{p}" for p in PRODUCTS]
CURRENT_PRICE_FEATURES = [f"log_price_current_{p}" for p in PRODUCTS]
CURRENT_PROMOTION_FEATURES = [f"promotion_current_{p}" for p in PRODUCTS]
BASELINE_MODELS = ["last_week", "mean_four_weeks", "seasonal_52_weeks"]


def calendar_lag(panel, columns, lag):
    """Retrieve exactly t-lag in the same store, never the previous available row."""
    shifted = panel[["store", "week"] + columns].copy()
    shifted["week"] += lag
    return (
        shifted.set_index(["store", "week"])[columns]
        .reindex(pd.MultiIndex.from_frame(panel[["store", "week"]]))
        .set_axis(panel.index)
    )


def construct_forecast_data(panel):
    """Build common target rows and source-dated features without fitting anything.

    Return a long panel. Require four consecutive sales weeks after a 52-week
    warmup. An absent annual lag remains missing for training-only imputation;
    the seasonal benchmark then falls back to the available four-week average.
    Missing complete-week traffic remains missing until training-only imputation.
    """
    panel = panel.sort_values(["store", "week"]).reset_index(drop=True)
    quantities = [f"units_{p}" for p in PRODUCTS]
    one = calendar_lag(
        panel, quantities + PRICE_NAMES + PROMOTION_NAMES + ["customers"], 1
    )
    annual = calendar_lag(panel, quantities, 52)
    recent = [calendar_lag(panel, quantities, lag) for lag in range(1, 5)]
    recent_mean = sum(recent) / 4
    common = one[quantities].notna().all(axis=1)
    common &= recent_mean.notna().all(axis=1)
    outputs = []
    for product in PRODUCTS:
        data = panel[["store", "week", "income"]].copy()
        data["product"] = product
        data["target"] = panel[f"units_{product}"]
        data["lag1"] = one[f"units_{product}"]
        data["mean4"] = recent_mean[f"units_{product}"]
        data["lag52"] = annual[f"units_{product}"]
        data["seasonal_fallback"] = data.lag52.isna()
        data["log_lag1"] = np.log1p(data.lag1)
        data["log_mean4"] = np.log1p(data.mean4)
        data["log_lag52"] = np.log1p(data.lag52)
        other = one[quantities].sum(axis=1) - data.lag1
        data["log_other_lag1"] = np.log1p(other)
        data["log_traffic_lag1"] = np.log(one.customers)
        data[LAG_PRICE_FEATURES] = np.log(one[PRICE_NAMES]).to_numpy()
        data[LAG_PROMOTION_FEATURES] = one[PROMOTION_NAMES].to_numpy()
        data[CURRENT_PRICE_FEATURES] = np.log(panel[PRICE_NAMES]).to_numpy()
        data[CURRENT_PROMOTION_FEATURES] = panel[PROMOTION_NAMES].to_numpy()
        data["season_sin"] = np.sin(2 * np.pi * (data.week - 1) / 52.1775)
        data["season_cos"] = np.cos(2 * np.pi * (data.week - 1) / 52.1775)
        data["trend"] = data.week / 52.1775
        data["last_sales_information_week"] = data.week - 1
        data["price_information_week_operational"] = data.week - 1
        data["price_information_week_conditional"] = data.week
        data["actual_price"] = panel[f"price_{product}"]
        data["last_price"] = one[f"price_{product}"]
        data["diagnostic_promotion"] = panel[f"promotion_{product}"]
        outputs.append(data[common & data.week.ge(53) & data.target.notna()])
    result = (
        pd.concat(outputs, ignore_index=True)
        .sort_values(["week", "store", "product"])
        .reset_index(drop=True)
    )
    assert result.target.ge(0).all()
    assert result.last_sales_information_week.lt(result.week).all()
    return result


def feature_names(information_set):
    """List the permitted regressors explicitly for each forecast information set."""
    if information_set == "operational_full_category":
        numeric = [
            "log_lag1",
            "log_mean4",
            "log_lag52",
            "log_traffic_lag1",
            "income",
            "season_sin",
            "season_cos",
            "trend",
            "log_avg_package_price_lag1",
            "log_active_upcs_lag1",
        ]
        return ["store", "product"] + numeric, numeric
    numeric = NUMERIC_FEATURES + LAG_PRICE_FEATURES + LAG_PROMOTION_FEATURES
    if information_set == "conditional_current_policy":
        numeric = numeric + CURRENT_PRICE_FEATURES + CURRENT_PROMOTION_FEATURES
    return ["store", "product"] + numeric, numeric


def candidate_models(config):
    """Return a small economically useful comparison, not a broad tuning search."""
    candidates = [(name, name, 0) for name in BASELINE_MODELS]
    candidates += [
        (f"ridge_{alpha:g}", "ridge", alpha)
        for alpha in config["forecast"]["ridge_alphas"]
    ]
    candidates += [
        (f"forest_{leaves}", "forest", leaves)
        for leaves in config["forecast"]["forest_leaf_counts"]
    ]
    candidates += [
        (f"boosting_{leaves}", "boosting", leaves)
        for leaves in config["forecast"]["boosting_leaf_counts"]
    ]
    candidates += [
        (
            "ridge_calibrated",
            "ridge_calibrated",
            config["forecast"]["calibrated_ridge_alpha"],
        )
    ]
    return candidates


def fit_forecast_model(training, family, parameter, information_set, config):
    """Fit the named family/parameter on past training rows for information_set.

    config supplies tuning, calibration and reproducibility settings. Return
    fitted preprocessing/model objects, product smearing and fit-origin records.
    """
    if family == "ridge_calibrated":
        return fit_calibrated_ridge(training, parameter, information_set, config)
    columns, numeric = feature_names(information_set)
    preprocessing = ColumnTransformer(
        [
            (
                "store_product",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["store", "product"],
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )
    settings = config["forecast"]
    if family == "ridge":
        estimator = Ridge(alpha=parameter)
    elif family == "forest":
        estimator = RandomForestRegressor(
            n_estimators=settings["forest_trees"],
            max_leaf_nodes=int(parameter),
            min_samples_leaf=30,
            max_features=0.6,
            n_jobs=settings["threads"],
            random_state=settings["random_seed"],
        )
    elif family == "boosting":
        # Disable random internal holdout selection; all tuning is chronological.
        estimator = HistGradientBoostingRegressor(
            max_iter=settings["boosting_iterations"],
            max_leaf_nodes=int(parameter),
            learning_rate=0.08,
            min_samples_leaf=30,
            early_stopping=False,
            random_state=settings["random_seed"],
        )
    else:
        raise ValueError(f"Unknown learned model {family}")
    model = Pipeline([("preprocessing", preprocessing), ("estimator", estimator)])
    logarithmic = family in ["ridge", "boosting"]
    target = np.log1p(training.target) if logarithmic else training.target
    model.fit(training[columns], target)
    if logarithmic:
        residual = target.to_numpy() - model.predict(training[columns])
        smearing = (
            pd.Series(np.exp(residual))
            .groupby(training["product"].to_numpy())
            .mean()
            .to_dict()
        )
    else:
        smearing = {p: 1.0 for p in training["product"].unique()}
    return dict(
        model=model,
        columns=columns,
        logarithmic=logarithmic,
        smearing=smearing,
        train_through=int(training.week.max()),
        training_rows=len(training),
        family=family,
        parameter=parameter,
        information_set=information_set,
    )


def fit_calibrated_ridge(training, alpha, information_set, config):
    """Fit ridge and estimate bias from prior held-out one-step predictions.

    The last configured weeks within training form a calibration tail. Both
    the preliminary fit and its smearing use only the earlier prefix. Factors
    are frozen for the next block; the final ridge refit uses all past data.
    """
    cutoff = int(training.week.max()) - config["forecast"]["point_calibration_weeks"]
    prefix = training[training.week.le(cutoff)]
    tail = training[training.week.gt(cutoff)]
    preliminary = fit_forecast_model(prefix, "ridge", alpha, information_set, config)
    prediction = predict_forecast(preliminary, tail)
    tail = tail.assign(calibration_prediction=prediction)
    factors = {}
    for product, group in tail.groupby("product"):
        raw = group.target.sum() / group.calibration_prediction.sum()
        weeks = group.week.nunique()
        weight = weeks / (weeks + config["forecast"]["calibration_shrinkage_weeks"])
        factors[product] = float(
            np.clip(1 + weight * (raw - 1), *config["forecast"]["factor_bounds"])
        )
    final = fit_forecast_model(training, "ridge", alpha, information_set, config)
    final["point_calibration"] = factors
    final["point_calibration_first"] = cutoff + 1
    final["point_calibration_last"] = int(tail.week.max())
    final["point_calibration_prefix_last"] = preliminary["train_through"]
    final["family"] = "ridge_calibrated"
    return final


def predict_forecast(model, data):
    """Predict mean package sales; exponentiate first, then subtract the log1p shift."""
    prediction = model["model"].predict(data[model["columns"]])
    if model["logarithmic"]:
        factors = data["product"].map(model["smearing"]).to_numpy()
        prediction = np.exp(prediction) * factors - 1
    prediction = np.maximum(prediction, 0)
    factors = model.get("point_calibration", {})
    return prediction * data["product"].map(factors).fillna(1).to_numpy()


def baseline_forecast(data, family):
    """Return a persistence, moving-average or seasonal forecast using dated lags."""
    source = {
        "last_week": "lag1",
        "mean_four_weeks": "mean4",
        "seasonal_52_weeks": "lag52",
    }[family]
    if family == "seasonal_52_weeks":
        return data.lag52.fillna(data.mean4).to_numpy()
    return data[source].to_numpy()


def accuracy(actual, predicted):
    """Return original-scale errors; WAPE remains well-defined at zero individual sales."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = predicted - actual
    return dict(
        n=len(actual),
        MAE=np.mean(abs(error)),
        RMSE=np.sqrt(np.mean(error**2)),
        WAPE=np.sum(abs(error)) / np.sum(actual),
        bias=np.mean(error),
        aggregate_ratio=np.sum(predicted) / np.sum(actual),
    )


def prediction_frame(
    data, prediction, category, information_set, model_id, family, phase, train_through
):
    """Return the shared schema for every forecast, including estimator cutoff."""
    result = data[
        [
            "store",
            "week",
            "product",
            "target",
            "actual_price",
            "last_price",
            "seasonal_fallback",
        ]
    ].copy()
    result["prediction"] = prediction
    result["evaluation"] = "retrospective"
    result["category"] = category
    result["information_set"] = information_set
    result["model_id"] = model_id
    result["family"] = family
    result["phase"] = phase
    result["train_through"] = train_through
    assert (result.train_through < result.week).all()
    result["predicted_revenue"] = prediction * (
        result.actual_price
        if information_set == "conditional_current_policy"
        else result.last_price
    )
    if "diagnostic_promotion" in data:
        result["diagnostic_promotion"] = data.diagnostic_promotion
    return result


def validate_candidates(data, category, information_set, config, logger):
    """Compare every candidate on two expanding-window validation blocks."""
    sample = config["sample"]
    folds = list(
        range(
            sample["train_end"] + 1,
            sample["selection_end"] + 1,
            config["forecast"]["validation_fold_weeks"],
        )
    )
    predictions, fit_records = [], []
    for model_id, family, parameter in candidate_models(config):
        for first_week in folds:
            last_week = min(
                first_week + config["forecast"]["validation_fold_weeks"] - 1,
                sample["selection_end"],
            )
            training = data[data.week < first_week]
            validation = data[data.week.between(first_week, last_week)]
            assert len(training) > 0
            if len(validation) == 0:
                # A source-data gap leaves the first cereal validation window
                # without dated-lag targets. Its absence is separately audited.
                continue
            if family in BASELINE_MODELS:
                predicted = baseline_forecast(validation, family)
                train_through = first_week - 1
            else:
                model = fit_forecast_model(
                    training, family, parameter, information_set, config
                )
                predicted = predict_forecast(model, validation)
                train_through = model["train_through"]
                fit_records.extend(
                    fit_record_rows(model, category, model_id, "validation")
                )
            predictions.append(
                prediction_frame(
                    validation,
                    predicted,
                    category,
                    information_set,
                    model_id,
                    family,
                    "validation",
                    train_through,
                )
            )
        logger.info(
            "%s %s validation finished: %s", category, information_set, model_id
        )
    predictions = pd.concat(predictions, ignore_index=True)
    scores = []
    for model_id, group in predictions.groupby("model_id", sort=False):
        scores.append(
            dict(
                category=category,
                information_set=information_set,
                model_id=model_id,
                family=group.family.iloc[0],
                **accuracy(group.target, group.prediction),
            )
        )
    scores = pd.DataFrame(scores).sort_values("WAPE")
    chosen = scores.iloc[0].model_id
    # Carry each model family's best validation setting into the test comparison.
    finalists = scores.groupby("family", sort=False).first().model_id.tolist()
    parameters = {
        name: (family, parameter)
        for name, family, parameter in candidate_models(config)
    }
    return predictions, scores, chosen, finalists, parameters, fit_records


def fit_record_rows(model, category, model_id, phase):
    """Record every fitted information set and product-specific smearing factor."""
    return [
        dict(
            category=category,
            information_set=model["information_set"],
            model_id=model_id,
            phase=phase,
            train_through=model["train_through"],
            training_rows=model["training_rows"],
            product=product,
            smearing_factor=factor,
            logarithmic=model["logarithmic"],
            point_factor=model.get("point_calibration", {}).get(product, 1.0),
            point_calibration_first=model.get("point_calibration_first", np.nan),
            point_calibration_last=model.get("point_calibration_last", np.nan),
            point_calibration_prefix_last=model.get(
                "point_calibration_prefix_last", np.nan
            ),
            feature_columns=";".join(model["columns"]),
        )
        for product, factor in model["smearing"].items()
    ]


def interval_radius(scores, coverage):
    """Return the ceil((n+1)*coverage)-th empirical score, capped at n.

    The finite-sample rank is explicit. Dependent scanner errors still prevent
    a claim of distribution-free prospective coverage.
    """
    scores = np.asarray(scores, float)
    assert len(scores) > 0 and np.isfinite(scores).all()
    rank = min(int(np.ceil((len(scores) + 1) * coverage)), len(scores))
    return float(np.sort(scores)[rank - 1])


def calibrate_intervals(
    data, family, parameter, information_set, category, model_id, config
):
    """Calibrate relative absolute forecast errors on a separate pretest period."""
    sample = config["sample"]
    training = data[data.week.le(sample["selection_end"])]
    calibration = data[
        data.week.between(sample["selection_end"] + 1, sample["calibration_end"])
    ]
    if family in BASELINE_MODELS:
        prediction = baseline_forecast(calibration, family)
        records = []
    else:
        model = fit_forecast_model(training, family, parameter, information_set, config)
        prediction = predict_forecast(model, calibration)
        records = fit_record_rows(model, category, model_id, "calibration")
    absolute = abs(calibration.target.to_numpy() - prediction) / (1 + prediction)
    factors = {}
    rows = []
    for product in calibration["product"].unique():
        mask = calibration["product"].eq(product).to_numpy()
        n = int(mask.sum())
        assert n > 0
        factor = interval_radius(
            absolute[mask], config["forecast"]["interval_coverage"]
        )
        factors[product] = factor
        rows.append(
            dict(
                category=category,
                information_set=information_set,
                model_id=model_id,
                product=product,
                calibration_first=sample["selection_end"] + 1,
                calibration_last=sample["calibration_end"],
                n=n,
                relative_error_radius=factor,
                nominal_coverage=config["forecast"]["interval_coverage"],
            )
        )
    return factors, rows, records


def rolling_test(
    data, category, information_set, model_id, family, parameter, factors, config
):
    """Refit at preannounced test origins; only earlier realized test rows can enter."""
    start = config["sample"]["calibration_end"] + 1
    end = config["sample"]["last_week"]
    outputs, records = [], []
    for first_week in range(start, end + 1, config["forecast"]["refit_every_weeks"]):
        last_week = min(first_week + config["forecast"]["refit_every_weeks"] - 1, end)
        training = data[data.week < first_week]
        test = data[data.week.between(first_week, last_week)]
        if len(test) == 0:
            continue
        if family in BASELINE_MODELS:
            prediction = baseline_forecast(test, family)
            train_through = first_week - 1
            model = None
        else:
            model = fit_forecast_model(
                training, family, parameter, information_set, config
            )
            prediction = predict_forecast(model, test)
            train_through = model["train_through"]
            records.extend(fit_record_rows(model, category, model_id, "test"))
        result = prediction_frame(
            test,
            prediction,
            category,
            information_set,
            model_id,
            family,
            "test",
            train_through,
        )
        radius = result["product"].map(factors).to_numpy() * (1 + prediction)
        result["lower"] = np.maximum(prediction - radius, 0)
        result["upper"] = prediction + radius
        result["covered"] = result.target.between(result.lower, result.upper)
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True), records, model


def construct_category_forecast_data(config, category):
    """Forecast complete-category package movement, allowing changing UPC assortments.

    Package counts are a retail volume measure, not liters, ounces or individual
    choices. The average package price is a lagged predictive covariate only;
    it is not used as a demand-system price or interpreted as an elasticity.
    """
    processed = config["paths"]["processed"]
    totals = pd.read_parquet(processed / f"category_weekly_{category}.parquet")
    traffic = pd.read_parquet(processed / "weekly_customers.parquet")
    demo = pd.read_parquet(processed / "store_demographics.parquet")
    panel = totals.merge(
        traffic, on=["store", "week"], how="left", validate="one_to_one"
    )
    panel = panel.merge(
        demo[["store", "income"]], on="store", how="left", validate="many_to_one"
    )
    panel = panel.sort_values(["store", "week"]).reset_index(drop=True)
    panel["average_package_price"] = (
        panel.category_revenue / panel.priced_units.replace(0, np.nan)
    )
    columns = ["category_units", "customers", "average_package_price", "active_upcs"]
    one = calendar_lag(panel, columns, 1)
    recent = [calendar_lag(panel, ["category_units"], lag) for lag in range(1, 5)]
    mean = sum(recent) / 4
    annual = calendar_lag(panel, ["category_units"], 52)
    data = panel[["store", "week", "income"]].copy()
    data["product"] = "all_upcs"
    data["target"] = panel.category_units
    data["lag1"] = one.category_units
    data["mean4"] = mean.category_units
    data["lag52"] = annual.category_units
    data["seasonal_fallback"] = data.lag52.isna()
    data["log_lag1"] = np.log1p(data.lag1)
    data["log_mean4"] = np.log1p(data.mean4)
    data["log_lag52"] = np.log1p(data.lag52)
    data["log_traffic_lag1"] = np.log(one.customers)
    data["log_avg_package_price_lag1"] = np.log(one.average_package_price)
    data["log_active_upcs_lag1"] = np.log1p(one.active_upcs)
    data["season_sin"] = np.sin(2 * np.pi * (data.week - 1) / 52.1775)
    data["season_cos"] = np.cos(2 * np.pi * (data.week - 1) / 52.1775)
    data["trend"] = data.week / 52.1775
    data["actual_price"] = panel.average_package_price
    data["last_price"] = one.average_package_price
    data["last_sales_information_week"] = data.week - 1
    data["price_information_week_operational"] = data.week - 1
    data = data[data[["lag1", "mean4"]].notna().all(axis=1) & data.week.ge(53)]
    assert (data.last_sales_information_week < data.week).all()
    return data.reset_index(drop=True)


def forecast_category(config, category, logger):
    """Execute the forecast targets for one category and return their result tables."""
    (
        validation_all,
        test_all,
        scores_all,
        selection_all,
        calibration_all,
        fits_all,
        audit_all,
        windows,
    ) = ([], [], [], [], [], [], [], [])
    panel = pd.read_parquet(config["paths"]["processed"] / f"panel_{category}.parquet")
    selected_data = construct_forecast_data(panel)
    selected_data.to_parquet(
        config["paths"]["processed"] / f"forecast_features_{category}.parquet",
        index=False,
    )
    category_data = construct_category_forecast_data(config, category)
    category_data.to_parquet(
        config["paths"]["processed"] / f"forecast_category_features_{category}.parquet",
        index=False,
    )
    exercises = [("operational_full_category", category_data)]
    calibration_rows = selected_data.week.between(
        config["sample"]["selection_end"] + 1, config["sample"]["calibration_end"]
    ).sum()
    test_rows = selected_data.week.gt(config["sample"]["calibration_end"]).sum()
    if calibration_rows > 0 and test_rows > 0:
        exercises.extend(
            [
                ("operational", selected_data),
                ("conditional_current_policy", selected_data),
            ]
        )
    else:
        logger.info(
            "%s selected UPCs have no calibration/test sample; full-category forecast executed",
            category,
        )
        audit_all.append(
            dict(
                category=category,
                information_set="selected_products_not_estimable",
                rows=len(selected_data),
                first_week=selected_data.week.min(),
                last_week=selected_data.week.max(),
                interpretation="Common dated-lag sample unavailable during calibration or test; individual UPCs can remain observed. Full-category forecasts executed; missing sales are not zero.",
            )
        )
    for information_set, data in exercises:
        for first_week in range(
            config["sample"]["train_end"] + 1,
            config["sample"]["selection_end"] + 1,
            config["forecast"]["validation_fold_weeks"],
        ):
            last_week = min(
                first_week + config["forecast"]["validation_fold_weeks"] - 1,
                config["sample"]["selection_end"],
            )
            count = int(data.week.between(first_week, last_week).sum())
            windows.append(
                dict(
                    category=category,
                    information_set=information_set,
                    first_week=first_week,
                    last_week=last_week,
                    validation_rows=count,
                    status="executed" if count else "not_estimable_no_observations",
                )
            )
            if count == 0:
                logger.info(
                    "%s %s validation weeks %s-%s have no eligible rows; recorded in forecast_validation_windows",
                    category,
                    information_set,
                    first_week,
                    last_week,
                )
        validation, scores, chosen, finalists, parameters, fit_records = (
            validate_candidates(data, category, information_set, config, logger)
        )
        validation_all.append(validation)
        scores_all.append(scores)
        fits_all.extend(fit_records)
        selection_all.append(
            dict(
                category=category,
                information_set=information_set,
                selected_model=chosen,
                selection_metric="validation WAPE",
                selection_last_week=config["sample"]["selection_end"],
                calibration_last_week=config["sample"]["calibration_end"],
                test_first_week=config["sample"]["calibration_end"] + 1,
            )
        )
        for model_id in finalists:
            family, parameter = parameters[model_id]
            factors, calibration, fit_records = calibrate_intervals(
                data, family, parameter, information_set, category, model_id, config
            )
            calibration_all.extend(calibration)
            fits_all.extend(fit_records)
            test, fit_records, last_model = rolling_test(
                data,
                category,
                information_set,
                model_id,
                family,
                parameter,
                factors,
                config,
            )
            test["selected_before_test"] = model_id == chosen
            test_all.append(test)
            fits_all.extend(fit_records)
            if model_id == chosen:
                model_file = (
                    config["paths"]["output"]
                    / "models"
                    / f"forecast_{category}_{information_set}.pkl"
                )
                if last_model is None:
                    model_file.unlink(missing_ok=True)
                else:
                    with open(model_file, "wb") as stream:
                        pickle.dump(last_model, stream)
            logger.info(
                "%s %s test %s: WAPE %.3f, interval coverage %.3f",
                category,
                information_set,
                model_id,
                accuracy(test.target, test.prediction)["WAPE"],
                test.covered.mean(),
            )
        audit_all.append(
            dict(
                category=category,
                information_set=information_set,
                rows=len(data),
                first_week=data.week.min(),
                last_week=data.week.max(),
                missing_annual_lag_targets=int(data.seasonal_fallback.sum()),
                quantities_information="t-1, t-2, t-3, t-4 and t-52",
                traffic_information="complete weekly count at t-1; missing values imputed using training only",
                price_information=(
                    "realized t prices and recorded promotions"
                    if information_set == "conditional_current_policy"
                    else "t-1"
                ),
                interpretation=(
                    "retrospective conditional forecast; historical plans were not supplied"
                    if information_set == "conditional_current_policy"
                    else "retrospective one-week-ahead operational forecast"
                ),
            )
        )
    return dict(
        validation=validation_all,
        test=test_all,
        scores=scores_all,
        selection=selection_all,
        calibration=calibration_all,
        fits=fits_all,
        audit=audit_all,
        windows=windows,
    )


def run_operational_forecasts(config):
    """Run each category, then assemble consistent forecast tables and accuracy."""
    logger = stage_log(config, "forecasting")
    results = [
        forecast_category(config, category, logger) for category in config["categories"]
    ]
    validation_all = [table for result in results for table in result["validation"]]
    test_all = [table for result in results for table in result["test"]]
    scores_all = [table for result in results for table in result["scores"]]
    selection_all = [row for result in results for row in result["selection"]]
    calibration_all = [row for result in results for row in result["calibration"]]
    fits_all = [row for result in results for row in result["fits"]]
    audit_all = [row for result in results for row in result["audit"]]
    windows = [row for result in results for row in result["windows"]]
    validation = pd.concat(validation_all, ignore_index=True)
    test = pd.concat(test_all, ignore_index=True)
    validation.to_parquet(
        config["paths"]["output"] / "validation_predictions.parquet", index=False
    )
    test.to_parquet(config["paths"]["output"] / "test_predictions.parquet", index=False)
    pd.concat(scores_all).to_csv(
        table_path(config, "forecast_validation_scores"), index=False
    )
    pd.DataFrame(selection_all).to_csv(
        table_path(config, "forecast_selection"), index=False
    )
    pd.DataFrame(calibration_all).to_csv(
        table_path(config, "forecast_interval_calibration"), index=False
    )
    pd.DataFrame(fits_all).to_csv(
        table_path(config, "forecast_fit_inventory"), index=False
    )
    pd.DataFrame(audit_all).to_csv(
        table_path(config, "forecast_information_sets"), index=False
    )
    pd.DataFrame(windows).to_csv(
        table_path(config, "forecast_validation_windows"), index=False
    )
    metrics = []
    groups = [
        "category",
        "information_set",
        "model_id",
        "product",
        "selected_before_test",
    ]
    for keys, group in test.groupby(groups):
        metrics.append(
            dict(
                zip(groups, keys),
                **accuracy(group.target, group.prediction),
                interval_coverage=group.covered.mean(),
                interval_width=(group.upper - group.lower).mean(),
            )
        )
    pd.DataFrame(metrics).to_csv(
        table_path(config, "forecast_test_product_metrics"), index=False
    )
    overall = []
    for keys, group in test.groupby(
        ["category", "information_set", "model_id", "selected_before_test"]
    ):
        overall.append(
            dict(
                zip(
                    ["category", "information_set", "model_id", "selected_before_test"],
                    keys,
                ),
                **accuracy(group.target, group.prediction),
                interval_coverage=group.covered.mean(),
                interval_width=(group.upper - group.lower).mean(),
            )
        )
    pd.DataFrame(overall).to_csv(
        table_path(config, "forecast_test_metrics"), index=False
    )
    weekly = test.groupby(
        ["category", "information_set", "model_id", "selected_before_test", "week"],
        as_index=False,
    )[["target", "prediction"]].sum()
    weekly.to_csv(table_path(config, "forecast_weekly_portfolio"), index=False)
    forecast_bias_groups(test, config)


def forecast_bias_groups(test, config):
    """Diagnose bias by product, promotion and price bins fixed before selection.

    Realized current prices/promotions are diagnostic labels, never operational
    regressors. Full-category price bins describe average package revenue mix.
    """
    rows = []
    grouped = ["category", "information_set", "model_id", "product"]
    for keys, group in test.groupby(grouped):
        category, information_set, model_id, product = keys
        if product == "all_upcs":
            training = pd.read_parquet(
                config["paths"]["processed"] / f"category_weekly_{category}.parquet"
            )
            training = training[training.week.le(config["sample"]["train_end"])]
            train_prices = (
                training.category_revenue / training.priced_units.replace(0, np.nan)
            ).dropna()
        else:
            training = pd.read_parquet(
                config["paths"]["processed"] / f"panel_{category}.parquet"
            )
            train_prices = training.loc[
                training.week.le(config["sample"]["train_end"]), f"price_{product}"
            ]
        low, high = np.quantile(train_prices, [1 / 3, 2 / 3])
        labels = np.select(
            [group.actual_price.le(low), group.actual_price.le(high)],
            ["low", "middle"],
            default="high",
        )
        scored = group.assign(training_price_bin=labels)
        dimensions = ["training_price_bin"]
        if product != "all_upcs":
            dimensions.append("diagnostic_promotion")
        partitions = [
            (dimension, label, subset)
            for dimension in dimensions
            for label, subset in scored.groupby(dimension)
        ]
        for dimension, label, subset in partitions:
            rows.append(
                dict(
                    zip(grouped, keys),
                    dimension=dimension,
                    group=label,
                    evaluation="retrospective",
                    **accuracy(subset.target, subset.prediction),
                )
            )
    pd.DataFrame(rows).to_csv(table_path(config, "forecast_bias_groups"), index=False)
