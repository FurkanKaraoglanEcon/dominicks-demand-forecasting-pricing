"""Small output helpers; no rounding enters estimation or model selection."""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def save_csv(data, directory, name):
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    Path(directory).mkdir(parents=True, exist_ok=True)
    frame.to_csv(Path(directory) / name, index=False, float_format="%.12g")
    return frame


def save_json(value, path):
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
    clean = frame.copy()
    for column in clean.select_dtypes("bool"):
        clean[column] = clean[column].astype("int8")
    clean.to_stata(path, write_index=False, version=118)


def calendar_lag(panel, column, lag):
    # Reindex actual calendar weeks: a previous record can be separated by a gap.
    keys = pd.MultiIndex.from_arrays([panel.store, panel.week - lag])
    return panel.set_index(["store", "week"])[column].reindex(keys).to_numpy()


SEASON = ["sin1", "cos1", "sin2", "cos2"]
PRICES = ["lp1", "lp2", "lp3", "lp4"]
PROMOTIONS = ["promo_flag1", "promo_flag2", "promo_flag3", "promo_flag4"]
