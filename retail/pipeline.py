"""One raw-to-report route; compatible tables are projections of the same results.

Inputs: the user-editable TOML configuration and original raw data.
Outputs: the complete stage sequence, timings, provenance manifest and economic report.
"""

import argparse
import hashlib
import importlib.metadata
import logging
import platform
import time
import tomllib
from pathlib import Path
from threadpoolctl import threadpool_limits
from .io import save_json
from .preparation import prepare
from .identification import price_audit
from .demand import estimate
from .category_panels import prepare_category_panels
from .demand_extensions import run_auxiliary_demand
from .pricing_forms import run_form_pricing
from .pricing_bootstrap import run_store_wild_sensitivity
from .verification_extensions import verify_extensions
from .pricing_scenarios import run_pricing
from .forecasting import run_forecasts
from .verification import verify
from .reporting import write_report


def load_config(path):
    """Read TOML at path and resolve data/output paths relative to that file.
    Assert essential settings, create generated directories and return the configuration.
    """
    path = Path(path).resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    for name, value in config["paths"].items():
        candidate = Path(value).expanduser()
        config["paths"][name] = (
            candidate if candidate.is_absolute() else path.parent / candidate
        )
    config["config_file"] = path
    for name in ["output", "reports", "processed"]:
        config["paths"][name].mkdir(parents=True, exist_ok=True)
    for name in ["tables", "models", "logs"]:
        (config["paths"]["output"] / name).mkdir(parents=True, exist_ok=True)
    assert (
        config["sample"]["train_end"]
        < config["sample"]["selection_end"]
        < config["sample"]["calibration_end"]
        < config["sample"]["last_week"]
    )
    assert config["data"]["products"] == 4
    assert 0 < config["data"]["training_share"] < 1
    assert 0 < config["data"]["validation_share"] < 1 - config["data"]["training_share"]
    assert config["demand"]["cluster"] == "store_and_week"
    assert 0 < config["identification"]["event_change"] < 1
    assert (
        0
        <= config["identification"]["event_rival_stability"]
        < config["identification"]["event_change"]
    )
    assert config["identification"]["event_window"] >= 1
    assert config["pricing"]["draws"] >= 100
    assert config["forecast"]["minimum_calibration_weeks"] >= 2
    return config


def main():
    """Read the command-line config path and execute every estimable raw-to-report stage.
    Write provenance/timings and exit unsuccessfully on failed verification."""
    parser = argparse.ArgumentParser(
        description="Retail pricing: complete raw-to-report pipeline"
    )
    parser.add_argument(
        "--config", default=Path(__file__).resolve().parents[1] / "config.toml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    # Bound BLAS/OpenMP parallelism for reproducible fitting and readable runtimes.
    threadpool_limits(limits=config["forecast"]["threads"])
    output = config["paths"]["output"]
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output / "analysis.log", mode="w"),
        ],
    )
    log = logging.getLogger(__name__)
    start = time.perf_counter()
    timings = {}

    def stage(name, function):
        log.info("%s", name)
        before = time.perf_counter()
        handler = logging.FileHandler(
            output / "logs" / f"stage_{name.split('.')[0]}.log", mode="w"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            value = function()
        except Exception:
            log.exception("Stage failed: %s", name)
            raise
        finally:
            root_logger.removeHandler(handler)
            handler.close()
        timings[name] = time.perf_counter() - before
        return value

    panel, selected = stage(
        "1. Complete raw preparation and training-only market selection",
        lambda: prepare(config),
    )
    stage(
        "1b. Visible category portfolios and operational panels",
        lambda: prepare_category_panels(panel, selected, config),
    )
    stage(
        "2. Price variation, isolated-event diagnostics and experimental-assignment audit",
        lambda: price_audit(panel, config),
    )
    models = stage(
        "3. Static, linked, omitted-substitute and dynamic demand",
        lambda: estimate(panel, config),
    )
    stage(
        "3b. Functional forms, income, visits, persistence, AAC IV and category responses",
        lambda: run_auxiliary_demand(config),
    )
    stage(
        "4. Historically constrained pricing and joint uncertainty",
        lambda: run_pricing(panel, models, config),
    )
    stage(
        "4b. Functional-form pricing and additional category portfolios",
        lambda: run_form_pricing(panel, models, config),
    )
    stage(
        "4c. Store-only wild bootstrap sensitivity",
        lambda: run_store_wild_sensitivity(config),
    )
    stage(
        "5. Chronological forecasts and out-of-sample calibration",
        lambda: run_forecasts(panel, config),
    )
    result = stage(
        "6. Sample, inference, information-set and economic verification",
        lambda: verify(config, panel, selected, models),
    )
    result = stage(
        "6b. Auxiliary inference and operational forecast verification",
        lambda: verify_extensions(config, result),
    )
    stage("7. Economic report and figures", lambda: write_report(config, result))
    save_json(
        dict(
            completed=True,
            from_raw=True,
            raw_categories=config["data"]["categories"],
            elapsed_seconds=time.perf_counter() - start,
            stage_seconds=timings,
            python=platform.python_version(),
            platform=platform.platform(),
            forecast_evaluation="retrospective",
            failed_checks=result["failed"],
            warning_checks=result["warnings"],
            dependencies={
                name: importlib.metadata.version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "matplotlib",
                    "reportlab",
                    "scikit-learn",
                    "pyarrow",
                    "threadpoolctl",
                ]
            },
            config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            source_sha256={
                str(
                    path.relative_to(Path(__file__).resolve().parents[1])
                ): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
            },
        ),
        output / "run_manifest.json",
    )
    log.info(
        "Completed in %.1f seconds: %s checks, %s failures, %s warnings",
        time.perf_counter() - start,
        result["checks"],
        result["failed"],
        result["warnings"],
    )
    if result["failed"]:
        raise SystemExit(2)
