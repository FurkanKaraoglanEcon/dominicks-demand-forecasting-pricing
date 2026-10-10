"""One forecast stage: retained conditional benchmarks and operational forecasts.

Inputs: prepared cola and category panels. Outputs: chronological candidate
scores, forecasts, calibration and information-set audits. The benchmark
exercise preserves the earlier research comparison; the operational exercise
uses t-1 prices/promotions and separate selection/calibration/test windows.
All windows reuse inspected historical data and are retrospective.
"""

from .forecast_benchmarks import run_benchmarks
from .forecast_operational import run_operational_forecasts


def run_forecasts(panel, config):
    """Execute both named forecast exercises from the same prepared raw inputs."""
    run_benchmarks(panel, config)
    run_operational_forecasts(config)
