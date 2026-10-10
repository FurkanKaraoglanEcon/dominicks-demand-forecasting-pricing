"""Small output helpers; no rounding enters estimation or model selection.

Inputs: frames, values, named paths and calendar keys.
Outputs: atomic tables, JSON/Stata datasets, calendar lags and stage loggers."""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def save_csv(data, directory, name):
    """Publish data as directory/name after writing a complete temporary CSV.
    Return the exported DataFrame; preserve numerical precision to 12 significant digits.
    """
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    Path(directory).mkdir(parents=True, exist_ok=True)
    target = Path(directory) / name
    temporary = target.with_name(target.name + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.12g")
    temporary.replace(target)
    return frame


def save_json(value, path):
    """Write value to path with NumPy/path conversion; return no data."""

    def convert(item):
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        if isinstance(item, (np.ndarray, pd.Index)):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=convert) + "\n")


def save_panel(frame, path):
    """Export frame to Stata path, converting Boolean columns; return no data."""
    clean = frame.copy()
    for column in clean.select_dtypes("bool"):
        clean[column] = clean[column].astype("int8")
    clean.to_stata(path, write_index=False, version=118)


def calendar_lag(panel, column, lag):
    # Reindex actual calendar weeks: a previous record can be separated by a gap.
    """Return column values at each panel store's week-minus-lag key.
    Absent calendar records produce NaN rather than the previous observed row."""
    keys = pd.MultiIndex.from_arrays([panel.store, panel.week - lag])
    return panel.set_index(["store", "week"])[column].reindex(keys).to_numpy()


SEASON = ["sin1", "cos1", "sin2", "cos2"]
PRICES = ["lp1", "lp2", "lp3", "lp4"]
PROMOTIONS = ["promo_flag1", "promo_flag2", "promo_flag3", "promo_flag4"]


def table_path(config, name):
    """Return a research table path, creating only its generated directory."""
    directory = config["paths"]["output"] / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.csv"


def stage_log(config, stage):
    """Use a dedicated stage file and the pipeline's terminal handler."""
    import logging

    directory = config["paths"]["output"] / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"retail.{stage}")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(directory / f"{stage}.log", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def split_code(week, train_end, validation_end):
    """Return training/validation/test codes 1/2/3 from week and the two endpoints."""
    return np.where(week <= train_end, 1, np.where(week <= validation_end, 2, 3))
