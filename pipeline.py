"""One raw-to-report route; compatible tables are projections of the same results."""

import argparse
import hashlib
import importlib.metadata
import logging
import platform
import time
import tomllib
from pathlib import Path
from .io import save_json
from .preparation import prepare
from .identification import price_audit
from .demand import estimate
from .pricing import run_pricing
from .forecasting import run_forecasts
from .verification import verify
from .reporting import write_report


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    for name, value in config["paths"].items():
        candidate = Path(value).expanduser()
        config["paths"][name] = (
            candidate if candidate.is_absolute() else path.parent / candidate
        )
    assert config["data"]["products"] == 4
    assert 0 < config["data"]["training_share"] < 1
    assert 0 < config["data"]["validation_share"] < 1 - config["data"]["training_share"]
    assert config["demand"]["cluster"] == "store_and_week"
    assert config["pricing"]["draws"] >= 100
    assert config["forecast"]["minimum_calibration_weeks"] >= 2
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Retail pricing: complete raw-to-report pipeline"
    )
    parser.add_argument(
        "--config", default=Path(__file__).resolve().parents[1] / "config.toml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
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
        value = function()
        timings[name] = time.perf_counter() - before
        return value

    panel, selected = stage(
        "1. Complete raw preparation and training-only market selection",
        lambda: prepare(config),
    )
    stage(
        "2. Independent price variation and experimental-assignment audit",
        lambda: price_audit(panel, config),
    )
    models = stage(
        "3. Static, linked, omitted-substitute and dynamic demand",
        lambda: estimate(panel, config),
    )
    stage(
        "4. Historically constrained pricing and joint uncertainty",
        lambda: run_pricing(panel, models, config),
    )
    stage(
        "5. Chronological forecasts and out-of-sample calibration",
        lambda: run_forecasts(panel, config),
    )
    result = stage(
        "6. Sample, inference, information-set and economic verification",
        lambda: verify(config, panel, selected, models),
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
                for name in ["numpy", "pandas", "scipy", "matplotlib", "reportlab"]
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
