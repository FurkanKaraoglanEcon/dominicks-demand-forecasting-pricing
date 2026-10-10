"""Describe identifying variation without relabeling observed prices as randomized assignments."""

import numpy as np
import pandas as pd
from .econometrics import absorb
from .io import PRICES, PROMOTIONS, save_csv, save_json, calendar_lag


def price_audit(panel, config):
    output = config["paths"]["output"] / "identification"
    sample = panel[PRICES + PROMOTIONS].notna().all(axis=1)
    data = panel.loc[sample]
    x = absorb(data[PRICES].to_numpy(), data.store, data.week)
    controls = absorb(data[PROMOTIONS].to_numpy(float), data.store, data.week)
    rows = []
    for j in range(4):
        other = np.column_stack([np.delete(x, j, axis=1), controls])
        residual = x[:, j] - other @ np.linalg.lstsq(other, x[:, j], rcond=None)[0]
        fraction = residual.var() / x[:, j].var()
        rows.append(
            dict(
                product=j + 1,
                raw_log_price_sd=data[PRICES[j]].std(),
                within_log_price_sd=x[:, j].std(),
                independent_log_price_sd=residual.std(),
                independent_variance_share=fraction,
                VIF=1 / fraction,
                unique_prices=data[f"p{j+1}"].nunique(),
            )
        )
    save_csv(rows, output, "independent_price_variation.csv")
    save_csv(
        pd.DataFrame(x).corr().rename_axis("product").reset_index(),
        output,
        "within_price_correlations.csv",
    )
    singular = np.linalg.svd(x / x.std(axis=0), compute_uv=False)
    pair_rows = []
    for family, pair in enumerate([(1, 2), (3, 4)], 1):
        difference = panel[f"lp{pair[0]}"] - panel[f"lp{pair[1]}"]
        changes = np.column_stack(
            [panel[f"lp{j}"] - calendar_lag(panel, f"lp{j}", 1) for j in pair]
        )
        available = np.isfinite(changes).all(axis=1)
        independent = (np.abs(changes[:, 0] - changes[:, 1]) > 0.01) & available
        isolated = (
            (np.max(np.abs(changes), axis=1) > 0.05)
            & (np.min(np.abs(changes), axis=1) < 0.01)
            & available
        )
        pair_rows.append(
            dict(
                family=family,
                equal_price_share=np.isclose(difference, 0, atol=1e-8).mean(),
                wedge_sd=difference.std(),
                available_changes=int(available.sum()),
                different_change_share=independent.sum() / available.sum(),
                isolated_changes=int(isolated.sum()),
            )
        )
    save_csv(pair_rows, output, "regular_diet_linkage.csv")
    zone = (
        panel.loc[panel.zone.notna()]
        .groupby(["zone", "week"])
        .agg(
            stores=("store", "nunique"),
            pepsi_prices=("p1", "nunique"),
            coke_prices=("p3", "nunique"),
        )
    )
    save_csv(zone.reset_index(), output, "zone_week_price_diversity.csv")
    save_json(
        dict(
            price_rank=int(np.linalg.matrix_rank(x)),
            condition_number=float(singular[0] / singular[-1]),
            assignment_columns_in_inputs=[],
            causal_design_available=False,
            actual_experiment_labels_available=False,
            observed_price_events_are_assignments=False,
            note="The codebook establishes experiments existed, not treatment membership of these rows.",
        ),
        output / "identification_audit.json",
    )
    text = """# Documented experiments and missing assignments

The Kilts Center describes randomized store-level research in more than 25 categories.
The uploaded movement columns are STORE, UPC, WEEK, MOVE, QTY, PRICE, SALE, PROFIT,
OK and two hexadecimal precision fields. UPC metadata, demographics and customer
counts provide no category-store treatment assignment or randomized price instrument.

Abrams, Gui and Hortacsu (2018), *Finding Exogenous Variation in Data*, discusses
1992-1993 EDLP/Hi-Lo/control experiments and includes soft drinks and cereals in its
recovery exercise. The authors had background documentation supplied separately
and report imperfect implementation. Their documented or recovered labels are not
contained in the attachments. Price regimes, SALE flags and socioeconomic zones are
not assignment labels. This project does not infer randomization from them.

A causal extension needs original store-category assignments, treatment dates,
affected UPCs, control stores, and implementation records. Accounting costs reconstructed
from retail margins and other-store prices are not automatically valid instruments.

Sources accessed 2026-10-09:
- https://www.chicagobooth.edu/research/kilts/research-data/dominicks
- https://arxiv.org/html/1704.07787v3 (Section 4 and Appendix C)
- Uploaded Dominick's manual, pages 3, 10-11 and 19.
"""
    (output / "experimental_assignment_audit.md").write_text(text)
