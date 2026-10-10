"""One-week-ahead rolling forecasts and calibration using earlier out-of-sample errors.

Inputs: prepared cola panel, forecast settings and output paths.
Outputs: eight retained forecast comparisons, activity sensitivity, calibration and bias audits.
"""

import logging
import numpy as np
import pandas as pd
from .io import (
    SEASON,
    PRICES,
    PROMOTIONS,
    save_csv,
    save_panel,
    save_json,
    calendar_lag,
)
from .econometrics import forecast_levels, metrics

LOG = logging.getLogger(__name__)


def features_for(product):
    """Return calendar/lag and current-policy regressor lists for the product slot."""
    calendar = SEASON + [f"lag1_{product}", f"lag52z_{product}"]
    pricing = calendar + PRICES + PROMOTIONS
    return calendar, pricing


def calibrate_origins(panel, product, base, settings, by_promotion):
    """Adjust base forecasts for product using prior panel errors and settings.
    by_promotion separates promotion states; return forecasts and origin-level factors.
    """
    prediction = base.copy()
    rows = []
    observed = panel[f"move{product}"].to_numpy(float)
    for origin in sorted(panel.loc[np.isfinite(base), "week"].unique()):
        past = (
            panel.week.lt(origin)
            & panel.week.ge(origin - settings["calibration_window"])
        ).to_numpy()
        past &= np.isfinite(base) & np.isfinite(observed) & (base > 0)
        current = panel.week.eq(origin).to_numpy() & np.isfinite(base)
        groups = [0, 1] if by_promotion else [-1]
        for promotion in groups:
            history = past.copy()
            target = current.copy()
            if promotion >= 0:
                history &= panel[f"promo_flag{product}"].eq(promotion).to_numpy()
                target &= panel[f"promo_flag{product}"].eq(promotion).to_numpy()
            if not target.any():
                continue
            weeks = panel.loc[history, "week"].nunique()
            if weeks >= settings["minimum_calibration_weeks"]:
                raw = observed[history].sum() / base[history].sum()
                weight = weeks / (weeks + settings["calibration_shrinkage_weeks"])
                factor = np.clip(1 + weight * (raw - 1), *settings["factor_bounds"])
            else:
                factor, raw = 1.0, np.nan
            prediction[target] = base[target] * factor
            rows.append(
                dict(
                    product=product,
                    origin=int(origin),
                    method="promo" if by_promotion else "pooled",
                    promotion=promotion,
                    factor=factor,
                    raw_factor=raw,
                    history_weeks=weeks,
                    history_rows=int(history.sum()),
                    calibration_max_week=(
                        int(panel.loc[history, "week"].max())
                        if history.any()
                        else np.nan
                    ),
                )
            )
    return prediction, rows


def predictions_for_product(panel, product, settings):
    """Use panel histories and settings to forecast one product.
    Return candidate arrays, fit-origin records and point-calibration records."""
    calendar, pricing = features_for(product)
    n = len(panel)
    predictions = {
        "seasonal_naive": panel[f"lag52_{product}"].to_numpy(float),
        "recent4": np.mean(
            np.column_stack(
                [calendar_lag(panel, f"move{product}", lag) for lag in range(1, 5)]
            ),
            axis=1,
        ),
    }
    logs = []
    train = panel.split.eq(1).to_numpy()
    targets = np.ones(n, bool)
    for name, features in [
        ("calendar", ["trend"] + calendar),
        ("pricing", ["trend"] + pricing),
        ("no_trend", pricing),
    ]:
        predicted, fit, smear = forecast_levels(
            panel, f"zq{product}", features, train, targets
        )
        predictions[name] = predicted
        logs.append(
            dict(
                product=product,
                model=name,
                origin=0,
                fit_first_week=int(panel.loc[fit.mask, "week"].min()),
                fit_last_week=int(panel.loc[fit.mask, "week"].max()),
                fit_rows=fit.n,
                smear=smear,
                input_information="training outcomes, store effects, calendar and prior sales; pricing variants assume current prices/promotions known",
            )
        )
    rolling = np.full(n, np.nan)
    activity = np.full(n, np.nan)
    train_end = int(panel.loc[train, "week"].max())
    first_origin = train_end - settings["calibration_warmup"] + 1
    activity_features = pricing + [
        "lag_activity_cer",
        "lag_activity_lnd",
        "lag_log_traffic",
    ]
    for origin in sorted(panel.loc[panel.week.ge(first_origin), "week"].unique()):
        fitting = (
            panel.week.lt(origin).to_numpy()
            & panel.week.ge(origin - settings["rolling_window"]).to_numpy()
        )
        target = panel.week.eq(origin).to_numpy()
        for name, features, storage in [
            ("rolling_no_trend", pricing, rolling),
            ("rolling_store_activity", activity_features, activity),
        ]:
            values, fit, smear = forecast_levels(
                panel, f"zq{product}", features, fitting, target
            )
            storage[target] = values
            logs.append(
                dict(
                    product=product,
                    model=name,
                    origin=int(origin),
                    fit_first_week=int(panel.loc[fit.mask, "week"].min()),
                    fit_last_week=int(panel.loc[fit.mask, "week"].max()),
                    fit_rows=fit.n,
                    smear=smear,
                    input_information="outcomes through origin-1; current posted prices/promotions assumed known; store traffic and other categories enter only lagged",
                )
            )
    predictions["rolling_no_trend"] = rolling
    predictions["rolling_store_activity"] = activity
    predictions["rolling_calibrated"], pooled = calibrate_origins(
        panel, product, rolling, settings, False
    )
    predictions["rolling_promo_calibrated"], promo = calibrate_origins(
        panel, product, rolling, settings, True
    )
    return predictions, logs, pooled + promo


def run_benchmarks(panel, config):
    """Run retained product forecasts using panel and config.
    Write metrics/predictions/audits, add chosen forecasts to panel and return choices.
    """
    settings = config["forecast"]
    output = config["paths"]["output"] / "forecast"
    (
        all_metrics,
        weekly,
        choices,
        origin_audit,
        calibration,
        diagnostics,
        coverage,
        activity_scores,
    ) = ([], [], [], [], [], [], [], [])
    for product in range(1, 5):
        (
            product_metrics,
            product_weekly,
            product_choices,
            product_origins,
            product_calibration,
            product_diagnostics,
            product_coverage,
            product_activity,
        ) = benchmark_product(panel, product, settings, output)
        all_metrics.extend(product_metrics)
        weekly.extend(product_weekly)
        choices.extend(product_choices)
        origin_audit.extend(product_origins)
        calibration.extend(product_calibration)
        diagnostics.extend(product_diagnostics)
        coverage.extend(product_coverage)
        activity_scores.extend(product_activity)
    metric_table = save_csv(all_metrics, output, "forecast_metrics_extended.csv")
    save_csv(metric_table, output, "forecast_metrics.csv")
    save_csv(weekly, output, "forecast_weekly_metrics.csv")
    save_csv(choices, output, "forecast_choices_extended.csv")
    save_csv(origin_audit, output, "forecast_origin_audit.csv")
    save_csv(calibration, output, "forecast_calibration_factors.csv")
    save_csv(diagnostics, output, "forecast_bias_diagnostics.csv")
    save_csv(coverage, output, "forecast_sample_coverage.csv")
    save_csv(activity_scores, output, "forecast_activity_sensitivity.csv")
    save_panel(panel, config["paths"]["output"] / "forecasts_panel.dta")
    save_json(
        dict(
            horizon="one calendar week",
            selection_period="retrospective_validation",
            final_evaluation_period="retrospective_test",
            fresh_untouched_test=False,
            level_prediction="max(0, exp(log1p prediction)*fitting-window smearing - 1)",
            calibration="trailing 26 weeks of prior one-step OOS forecasts; shrink toward 1 and clip to [0.5,2]",
            price_assumption="target-week retail prices and promotion schedules available at origin",
            customer_counts="lagged purchasing visits, never target-week traffic or household histories",
        ),
        output / "forecast_notes.json",
    )
    return choices


def score_benchmark_period(panel, product, model, prediction, period, split, sample):
    """Return promotion-split and weekly errors on one fixed scoring sample."""
    all_metrics, weekly = [], []
    validation_score = None
    for promotion in [-1, 0, 1]:
        selected = period.copy()
        if promotion >= 0:
            selected &= panel[f"promo_flag{product}"].eq(promotion).to_numpy()
        if not selected.any():
            continue
        score = metrics(panel.loc[selected, f"move{product}"], prediction[selected])
        all_metrics.append(
            dict(
                product=product,
                model=model,
                sample=sample,
                promotion=promotion,
                **score,
            )
        )
        if split == 2 and promotion == -1:
            validation_score = score["WAPE"]
    for week in sorted(panel.loc[period, "week"].unique()):
        selected = period & panel.week.eq(week).to_numpy()
        actual = panel.loc[selected, f"move{product}"].to_numpy()
        weekly.append(
            dict(
                product=product,
                model=model,
                sample=sample,
                week=int(week),
                actual=actual.sum(),
                predicted=prediction[selected].sum(),
                **metrics(actual, prediction[selected]),
            )
        )
    return all_metrics, weekly, validation_score


def benchmark_product(panel, product, settings, output):
    """Fit and score the retained conditional candidates for one product."""
    (
        all_metrics,
        weekly,
        choices,
        origin_audit,
        calibration,
        diagnostics,
        coverage,
        activity_scores,
    ) = ([], [], [], [], [], [], [], [])
    LOG.info("Chronological forecasting, product %s", product)
    predictions, logs, factors = predictions_for_product(panel, product, settings)
    origin_audit.extend(logs)
    calibration.extend(factors)
    common = panel.split.isin([2, 3]).to_numpy()
    for name in settings["models"]:
        common &= np.isfinite(predictions[name])
    activity_common = common & np.isfinite(predictions["rolling_store_activity"])
    coverage.append(
        dict(
            product=product,
            available_evaluation_rows=int(panel.split.isin([2, 3]).sum()),
            common_score_rows=int(common.sum()),
            activity_score_rows=int(activity_common.sum()),
            activity_information_set_rows_lost=int(
                common.sum() - activity_common.sum()
            ),
        )
    )
    # Supplemental activity signals have a smaller information-set sample.
    # Compare them with the base model on that same sample without shrinking the main evaluation.
    for model in ["rolling_no_trend", "rolling_store_activity"]:
        for split, sample in [
            (2, "retrospective_validation"),
            (3, "retrospective_test"),
        ]:
            sample_rows = activity_common & panel.split.eq(split).to_numpy()
            activity_scores.append(
                dict(
                    product=product,
                    model=model,
                    sample=sample,
                    **metrics(
                        panel.loc[sample_rows, f"move{product}"],
                        predictions[model][sample_rows],
                    ),
                )
            )
    validation_scores = {}
    for model in settings["models"]:
        prediction = predictions[model]
        for split, sample in [
            (2, "retrospective_validation"),
            (3, "retrospective_test"),
        ]:
            period = common & panel.split.eq(split).to_numpy()
            period_metrics, period_weekly, validation_score = score_benchmark_period(
                panel, product, model, prediction, period, split, sample
            )
            all_metrics.extend(period_metrics)
            weekly.extend(period_weekly)
            if validation_score is not None:
                validation_scores[model] = validation_score
    chosen = min(settings["models"], key=lambda model: validation_scores[model])
    choices.append(
        dict(
            product=product,
            model=chosen,
            validation_WAPE=validation_scores[chosen],
            selection_basis="configured validation weeks only; reused periods are retrospective; first configured model breaks exact ties",
        )
    )
    # Price bins are frozen using training prices, not future quantities or errors.
    low, high = np.quantile(panel.loc[panel.split.eq(1), f"p{product}"], [1 / 3, 2 / 3])
    price_bin = np.select(
        [panel[f"p{product}"].le(low), panel[f"p{product}"].le(high)],
        ["low", "middle"],
        default="high",
    )
    for model in dict.fromkeys(
        [
            "rolling_no_trend",
            "rolling_calibrated",
            "rolling_promo_calibrated",
            chosen,
        ]
    ):
        for group in ["low", "middle", "high"]:
            selected = common & panel.split.eq(3).to_numpy() & (price_bin == group)
            if selected.any():
                diagnostics.append(
                    dict(
                        product=product,
                        model=model,
                        dimension="training_price_bin",
                        group=group,
                        **metrics(
                            panel.loc[selected, f"move{product}"],
                            predictions[model][selected],
                        ),
                    )
                )
    exported = pd.DataFrame(
        dict(
            store=panel.store,
            week=panel.week,
            split=panel.split,
            actual=panel[f"move{product}"],
            common_score=common,
        )
    )
    for name, prediction in predictions.items():
        exported[name] = prediction
    exported["chosen_forecast"] = predictions[chosen]
    save_csv(exported, output, f"forecast_predictions_product{product}.csv")
    panel[f"forecast_{product}"] = predictions[chosen]
    return (
        all_metrics,
        weekly,
        choices,
        origin_audit,
        calibration,
        diagnostics,
        coverage,
        activity_scores,
    )
