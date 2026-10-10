"""Collect stage-specific economic verification with explicit numerical tolerances.

Inputs: current prepared samples, fitted systems and generated outputs.
Outputs: a readable check table and summary. No frozen reference is required.
"""

import numpy as np
from .io import save_csv, save_json
from .verification_research import (
    verify_raw_sample,
    verify_training_selection,
    verify_accounting_inference,
    verify_benchmark_forecasts,
    verify_primary_pricing,
)


def verify(config, panel, selected, models):
    """Verify the supplied panel, selected UPCs and fitted models under config tolerances.
    Export check rows and return the check-count/limitation summary."""
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

    audit = verify_raw_sample(config, panel, selected, models, check, close)
    weights = verify_training_selection(config, panel, selected, check, close)
    verify_accounting_inference(config, panel, models, check, close)
    verify_benchmark_forecasts(config, panel, audit, check, close)
    verify_primary_pricing(config, panel, models, weights, check, close)
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
