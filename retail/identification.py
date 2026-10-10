"""Describe identifying variation without relabeling observed prices as randomized assignments.

Inputs: prepared prices/promotions and configuration.
Outputs: partial variation, linkage, event profiles and an assignment-availability audit.
"""

import numpy as np
import pandas as pd
from .econometrics import absorb
from .io import PRICES, PROMOTIONS, save_csv, save_json, calendar_lag


def price_audit(panel, config):
    """Use panel prices/promotions and configured training/event definitions.
    Write identifying-variation and assignment-availability diagnostics; return no data.
    """
    output = config["paths"]["output"] / "identification"
    sample = panel[PRICES + PROMOTIONS].notna().all(axis=1)
    data = panel.loc[sample]
    x = absorb(data[PRICES].to_numpy(), data.store, data.week)
    controls = absorb(data[PROMOTIONS].to_numpy(float), data.store, data.week)
    rows = []
    legacy_rows = []
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
        price_controls = np.delete(x, j, axis=1)
        price_residual = (
            x[:, j]
            - price_controls @ np.linalg.lstsq(price_controls, x[:, j], rcond=None)[0]
        )
        legacy_rows.append(
            dict(
                product=j + 1,
                raw_sd=data[PRICES[j]].std(),
                residual_sd=x[:, j].std(ddof=1),
                variance_remaining=x[:, j].var() / data[PRICES[j]].var(ddof=0),
                price_only_VIF=x[:, j].var() / price_residual.var(),
            )
        )
    save_csv(rows, output, "independent_price_variation.csv")
    save_csv(legacy_rows, output, "price_variation.csv")
    diagnostics = []
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        diagnostics.append(
            dict(
                product1=a + 1,
                product2=b + 1,
                same_price=np.isclose(
                    data[f"p{a+1}"], data[f"p{b+1}"], atol=1e-6, rtol=0
                ).mean(),
                same_promo=data[PROMOTIONS[a]].eq(data[PROMOTIONS[b]]).mean(),
                raw_log_correlation=data[PRICES[a]].corr(data[PRICES[b]]),
                FE_log_correlation=np.corrcoef(x[:, a], x[:, b])[0, 1],
            )
        )
    save_csv(diagnostics, output, "price_pair_diagnostics.csv")
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
    price_events(panel, config)


def event_windows(panel, anchors, focal, rivals, any_event, window, stable):
    """Keep complete calendar windows; clean windows also exclude concurrent changes."""
    relative = np.arange(-window, window + 1)
    length = len(relative)
    event_id = np.repeat(np.arange(len(anchors)), length)
    stores = np.repeat(anchors.store.to_numpy(), length)
    weeks = np.repeat(anchors.week.to_numpy(), length) + np.tile(relative, len(anchors))
    indexed = panel.set_index(["store", "week"])
    columns = ["fq1", "fq2"] + PROMOTIONS + rivals
    keys = pd.MultiIndex.from_arrays([stores, weeks])
    values = indexed[columns].reindex(keys).reset_index(drop=True)
    base_keys = pd.MultiIndex.from_arrays([anchors.store, anchors.week - 1])
    base = indexed[columns].reindex(base_keys).reset_index(drop=True)
    repeated_base = base.iloc[event_id].reset_index(drop=True)
    present = indexed.index.get_indexer(keys) >= 0
    complete = pd.Series(present).groupby(event_id).all()
    complete &= base[["fq1", "fq2"]].gt(0).all(axis=1)
    same_promotions = values[PROMOTIONS].eq(repeated_base[PROMOTIONS]).all(axis=1)
    stable_rivals = (
        (values[rivals] - repeated_base[rivals]).abs().le(np.log1p(stable)).all(axis=1)
    )
    counts = pd.Series(any_event, index=indexed.index).reindex(keys).eq(True)
    event_days = pd.Series(counts.to_numpy()).groupby(event_id).sum()
    clean = (
        complete
        & pd.Series(same_promotions.to_numpy() & stable_rivals.to_numpy())
        .groupby(event_id)
        .all()
        & event_days.eq(1)
    )
    family = 1 if focal in ["flp1", "lp1", "lp2"] else 2
    rival_family = 3 - family
    frame = pd.DataFrame(
        dict(
            store=stores,
            anchor_week=np.repeat(anchors.week.to_numpy(), length),
            week=weeks,
            relative_week=np.tile(relative, len(anchors)),
            clean_window=clean.reindex(event_id).to_numpy(dtype=int),
            focal_family_pct=100
            * (values[f"fq{family}"] / repeated_base[f"fq{family}"] - 1),
            rival_family_pct=100
            * (values[f"fq{rival_family}"] / repeated_base[f"fq{rival_family}"] - 1),
        )
    )
    return (
        frame.loc[complete.reindex(event_id).to_numpy()],
        int(complete.sum()),
        int(clean.sum()),
    )


def price_events(panel, config):
    """Descriptive event screening preserves the earlier exercise without claiming randomization."""
    settings = config["identification"]
    output = config["paths"]["output"] / "identification"
    prices = ["flp1", "flp2"] + PRICES
    changes = {
        name: panel[name].to_numpy() - calendar_lag(panel, name, 1) for name in prices
    }
    promotion_stable = np.column_stack(
        [panel[name].to_numpy() == calendar_lag(panel, name, 1) for name in PROMOTIONS]
    ).all(axis=1)
    definitions, flags = [], []
    for focal in prices:
        rivals = (
            ["flp2" if focal == "flp1" else "flp1"]
            if focal.startswith("flp")
            else [p for p in PRICES if p != focal]
        )
        stable = np.column_stack(
            [
                np.abs(changes[p]) <= np.log1p(settings["event_rival_stability"])
                for p in rivals
            ]
        ).all(axis=1)
        for direction in [-1, 1]:
            own = (
                changes[focal] <= np.log1p(-settings["event_change"])
                if direction < 0
                else changes[focal] >= np.log1p(settings["event_change"])
            )
            name = focal + ("_cut" if direction < 0 else "_rise")
            definitions.append((focal, rivals, name))
            flags.append(own & stable & promotion_stable)
    any_event = np.column_stack(flags).any(axis=1)
    counts, stacks, anchor_tables = [], [], []
    for kind, ((focal, rivals, name), flag) in enumerate(zip(definitions, flags), 1):
        anchors = panel.loc[flag, ["store", "week"]]
        anchor_tables.append(
            anchors.assign(
                event_type=kind, event_name=name, focal_log_change=changes[focal][flag]
            )
        )
        complete, clean = 0, 0
        if len(anchors):
            rows, complete, clean = event_windows(
                panel,
                anchors,
                focal,
                rivals,
                any_event,
                settings["event_window"],
                settings["event_rival_stability"],
            )
            stacks.append(rows.assign(event_type=kind, event_name=name))
        counts.append(
            dict(
                event_type=kind,
                event_name=name,
                events=len(anchors),
                stores=anchors.store.nunique(),
                weeks=anchors.week.nunique(),
                complete_windows=complete,
                clean_windows=clean,
            )
        )
    save_csv(counts, output, "price_event_counts.csv")
    save_csv(
        pd.concat(anchor_tables, ignore_index=True), output, "price_event_anchors.csv"
    )
    columns = [
        "event_type",
        "event_name",
        "store",
        "anchor_week",
        "week",
        "relative_week",
        "clean_window",
        "focal_family_pct",
        "rival_family_pct",
    ]
    stack = (
        pd.concat(stacks, ignore_index=True)[columns]
        if stacks
        else pd.DataFrame(columns=columns)
    )
    save_csv(stack, output, "event_windows.csv")
    profiles = []
    for label, subset in [
        ("all_qualified_anchors", stack),
        ("strict_clean", stack.loc[stack.clean_window.eq(1)]),
    ]:
        if len(subset):
            profile = subset.groupby(
                ["event_type", "event_name", "relative_week"], as_index=False
            ).agg(
                focal_family_pct=("focal_family_pct", "mean"),
                rival_family_pct=("rival_family_pct", "mean"),
                events=("store", "size"),
            )
            profiles.append(profile.assign(sample=label))
    profile_columns = [
        "event_type",
        "event_name",
        "relative_week",
        "focal_family_pct",
        "rival_family_pct",
        "events",
        "sample",
    ]
    save_csv(
        (
            pd.concat(profiles, ignore_index=True)
            if profiles
            else pd.DataFrame(columns=profile_columns)
        ),
        output,
        "event_profiles.csv",
    )
