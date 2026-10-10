"""Research figures from generated tables; no estimation or frozen inputs.

Inputs: current run outputs. Outputs: coverage, dynamics, calibration, events
and observed sales figures, preserving earlier descriptive exercises.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = ["Pepsi", "Diet Pepsi", "Coca-Cola Classic", "Diet Coke"]
BLUE = "#173b58"
GREEN = "#168078"
ORANGE = "#c66b2b"


def make_figures(output, reports):
    """Read current tables in out and save coverage/demand/pricing figures in reports.
    Return no data; numerical tables remain the authoritative values."""
    directory = reports / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": BLUE,
            "text.color": BLUE,
            "axes.titleweight": "bold",
        }
    )
    coverage = pd.read_csv(output / "revenue_coverage.csv")
    fig, ax = plt.subplots(figsize=(7.8, 3))
    x = np.arange(3)
    ax.bar(
        x - 0.16,
        100 * coverage.market_coverage,
        width=0.32,
        label="Broader 2L cola",
        color=GREEN,
    )
    ax.bar(
        x + 0.16,
        100 * coverage.full_category_coverage,
        width=0.32,
        label="All soft drinks",
        color=BLUE,
    )
    ax.set(
        xticks=x,
        xticklabels=["Training", "Validation", "Later period"],
        ylim=(0, 100),
        ylabel="Selected revenue / market revenue (%)",
    )
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(directory / "market_coverage.png", dpi=180)
    plt.close(fig)
    dynamics = pd.read_csv(output / "demand/demand_dynamics.csv")
    own = dynamics.loc[
        dynamics.system.eq("linked_product")
        & dynamics.price.eq((dynamics.demand > 2).astype(int) + 1)
    ]
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    for offset, response, color in [
        (-0.18, "immediate", BLUE),
        (0, "subsequent", ORANGE),
        (0.18, "cumulative", GREEN),
    ]:
        frame = own.loc[own.response.eq(response)].sort_values("demand")
        ax.errorbar(
            x=np.arange(4) + offset,
            y=frame.elasticity,
            yerr=[frame.elasticity - frame.lower95, frame.upper95 - frame.elasticity],
            fmt="o",
            capsize=3,
            label=response,
            color=color,
        )
    ax.axhline(0, color="#aaaaaa", lw=0.8)
    ax.set(
        xticks=np.arange(4),
        xticklabels=["Pepsi", "Diet Pepsi", "Coke", "Diet Coke"],
        ylabel="Brand-price response (log points)",
    )
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(directory / "demand_dynamics.png", dpi=180)
    plt.close(fig)
    scores = pd.read_csv(output / "forecast/forecast_metrics_extended.csv")
    selected = scores.loc[
        scores["sample"].eq("retrospective_test") & scores.promotion.eq(-1)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for offset, model, label, color in [
        (-0.18, "rolling_no_trend", "Rolling", BLUE),
        (0, "rolling_calibrated", "Pooled calibration", ORANGE),
        (0.18, "rolling_promo_calibrated", "Promotion calibration", GREEN),
    ]:
        frame = selected.loc[selected.model.eq(model)].sort_values("product")
        for ax, column in zip(axes, ["WAPE", "bias_pct"]):
            ax.bar(
                np.arange(4) + offset,
                frame[column],
                width=0.18,
                label=label,
                color=color,
            )
    for ax in axes:
        ax.set(
            xticks=np.arange(4),
            xticklabels=["Pepsi", "Diet\nPepsi", "Coke", "Diet\nCoke"],
        )
        ax.axhline(0, color="#aaaaaa", lw=0.7)
    axes[0].set_ylabel("WAPE (%)")
    axes[1].set_ylabel("Aggregate bias (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "forecast_calibration.png", dpi=180)
    plt.close(fig)
    weekly = pd.read_csv(output / "forecast/forecast_weekly_metrics.csv")
    choices = pd.read_csv(output / "forecast/forecast_choices_extended.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for j, ax in enumerate(axes.ravel(), 1):
        chosen = choices.loc[choices["product"].eq(j), "model"].iloc[0]
        frame = weekly.loc[weekly["product"].eq(j) & weekly.model.eq(chosen)]
        ax.plot(frame.week, frame.actual, color=BLUE, label="Actual")
        ax.plot(frame.week, frame.predicted, color=GREEN, label="Selected forecast")
        ax.axvline(339.5, color=ORANGE, ls="--", lw=0.8)
        ax.set_title(NAMES[j - 1])
        ax.set_ylabel("Weekly units")
        ax.set_xlabel("Database week")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(directory / "selected_forecasts.png", dpi=180)
    plt.close(fig)
    descriptive_figures(output, directory)


def descriptive_figures(output, directory):
    """Read current descriptive tables in out and save calendar/event figures in reports.
    Return no data."""
    weekly = pd.read_csv(
        output / "weekly_segment_summary.csv", parse_dates=["start_date"]
    )
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(weekly.start_date, weekly.segment_units, color=BLUE)
    ax.set(
        xlabel="Week start",
        ylabel="Selected portfolio units",
        title="Observed weekly sales in the retained panel",
    )
    fig.tight_layout()
    fig.savefig(directory / "weekly_units.png", dpi=180)
    plt.close(fig)
    profiles = pd.read_csv(output / "identification/event_profiles.csv")
    titles = {
        1: "Pepsi family price cut",
        2: "Pepsi family price increase",
        3: "Coke family price cut",
        4: "Coke family price increase",
    }
    for kind, title in titles.items():
        frame = profiles.loc[
            profiles.event_type.eq(kind)
            & profiles["sample"].eq("all_qualified_anchors")
        ]
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        ax.plot(
            frame.relative_week,
            frame.focal_family_pct,
            label="Focal family",
            color=BLUE,
        )
        ax.plot(
            frame.relative_week,
            frame.rival_family_pct,
            label="Other family",
            color=GREEN,
        )
        ax.axvline(0, color=ORANGE, ls="--", lw=0.8)
        ax.axhline(0, color="#aaaaaa", lw=0.8)
        ax.set(
            xlabel="Weeks from price change",
            ylabel="Units relative to week -1 (%)",
            title=title + ": descriptive selected events",
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(directory / f"event_{kind}.png", dpi=180)
        plt.close(fig)
