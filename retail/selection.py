"""Training-only cola selection, omitted basket and common-price sample construction.

Inputs: valid movement and UPC definitions from preparation. Outputs: ranked
candidates, portfolio mappings, coverage tables and the balanced cola panel.
Missing prices and missing movement are never treated as zero demand.
"""

import numpy as np
from .io import save_csv, save_json, split_code


def select_products(movement, metadata, settings, train_end):
    """Return training-revenue-ranked comparable cola UPCs and candidate coverage. Future revenue never enters the ranking."""
    training = movement.loc[
        movement.week.le(train_end)
        & movement.upc.isin(metadata.loc[metadata.candidate, "upc"])
    ]
    denominator = len(training[["store", "week"]].drop_duplicates())
    candidates = training.groupby("upc", as_index=False).agg(
        train_revenue=("revenue", "sum"), observed_weeks=("move", "size")
    )
    candidates["coverage"] = candidates.observed_weeks / denominator
    qualified = candidates.loc[candidates.coverage.ge(settings["minimum_coverage"])]
    selected = qualified.sort_values(
        ["train_revenue", "upc"], ascending=[False, True]
    ).head(settings["products"])
    assert len(selected) == 4
    selected = selected.sort_values("upc").reset_index(drop=True)
    selected["product"] = np.arange(1, 5)
    selected = selected.merge(metadata, on="upc", validate="one_to_one")
    assert set(zip(selected.brand, selected.diet)) == {
        ("Pepsi", 0),
        ("Pepsi", 1),
        ("Coke", 0),
        ("Coke", 1),
    }
    assert selected.brand.tolist() == ["Pepsi", "Pepsi", "Coke", "Coke"]
    return selected, candidates


def omitted_index(movement, selected, train_end, settings, output):
    """Return a frozen training-weight geometric price index of observed omitted colas, with the changing observed-weight coverage exposed."""
    omitted = movement.loc[
        ~movement.upc.isin(selected.upc) & ~movement.ambiguous
    ].copy()
    training = omitted.loc[omitted.week.le(train_end)]
    denominator = (
        movement.loc[movement.week.le(train_end), ["store", "week"]]
        .drop_duplicates()
        .shape[0]
    )
    weights = training.groupby("upc", as_index=False).agg(
        train_revenue=("revenue", "sum"), observed_weeks=("move", "size")
    )
    weights["coverage"] = weights.observed_weeks / denominator
    weights = weights.loc[
        weights.coverage.ge(settings["omitted_minimum_coverage"])
    ].copy()
    assert not weights.empty, "No sufficiently observed omitted cola products"
    weights["weight"] = weights.train_revenue / weights.train_revenue.sum()
    save_csv(weights, output, "omitted_substitute_weights.csv")
    omitted = omitted.merge(
        weights[["upc", "weight"]], on="upc", validate="many_to_one"
    )
    omitted["weighted_log_price"] = omitted.weight * np.log(omitted.p)
    index = omitted.groupby(["store", "week"], as_index=False).agg(
        weighted_log_price=("weighted_log_price", "sum"),
        omitted_weight_coverage=("weight", "sum"),
    )
    # A changing observed basket is disclosed via weight coverage, not filled with zero prices.
    index["omitted_log_price"] = (
        index.weighted_log_price / index.omitted_weight_coverage
    )
    return index.drop(columns="weighted_log_price")


def build_panel(movement, selected, cutoffs, settings, output):
    """Return actual common four-price store-weeks and a restriction audit; retain stores by training history only."""
    long = movement.merge(
        selected[["upc", "product"]], on="upc", validate="many_to_one"
    )
    summary = long.groupby(["upc", "split", "product"], as_index=False).agg(
        observations=("move", "size"),
        units=("move", "sum"),
        revenue=("revenue", "sum"),
        grossprofit=("grossprofit", "sum"),
    )
    summary["split"] = summary.split.map({1: "Training", 2: "Validation", 3: "Test"})
    save_csv(summary, output, "product_sample_summary.csv")
    wide = long.pivot(
        index=["store", "week"],
        columns="product",
        values=["move", "p", "cost", "margin", "promo_flag"],
    )
    wide.columns = [f"{name}{item}" for name, item in wide.columns]
    panel = wide.reset_index()
    panel["nprice"] = panel[[f"p{j}" for j in range(1, 5)]].notna().sum(axis=1)
    panel["nquantity"] = panel[[f"move{j}" for j in range(1, 5)]].notna().sum(axis=1)
    panel["complete"] = panel.nprice.eq(4) & panel.nquantity.eq(4)
    audit = dict(
        markets_before_balance=len(panel), complete_markets=int(panel.complete.sum())
    )
    panel = panel.loc[panel.complete].copy()
    panel["split"] = split_code(panel.week, *cutoffs)
    histories = panel.loc[panel.split.eq(1)].groupby("store").size()
    stores = histories.loc[histories.ge(settings["minimum_training_weeks"])].index
    panel = (
        panel.loc[panel.store.isin(stores)]
        .sort_values(["store", "week"])
        .reset_index(drop=True)
    )
    assert panel.store.nunique() >= settings["minimum_stores"]
    assert not panel.duplicated(["store", "week"]).any()
    for j in range(1, 5):
        panel[f"lp{j}"] = np.log(panel[f"p{j}"])
        panel[f"lq{j}"] = np.log(panel[f"move{j}"].where(panel[f"move{j}"].gt(0)))
        panel[f"zq{j}"] = np.log1p(panel[f"move{j}"])
    audit.update(
        final_markets=len(panel),
        stores=panel.store.nunique(),
        train_end=cutoffs[0],
        validation_end=cutoffs[1],
        selection_last_week=cutoffs[0],
    )
    return panel, audit


def report_coverage(products, selected, output, settings):
    """Save selected/broader-cola and selected/all-soft-drink revenue shares, omissions and the explicit optimized-scope label."""
    rows = []
    for split, frame in products.groupby("split"):
        selected_revenue = frame.loc[frame.upc.isin(selected.upc), "revenue"].sum()
        market_revenue = frame.loc[frame.market_2l_cola, "revenue"].sum()
        rows.append(
            dict(
                split=split,
                selected_revenue=selected_revenue,
                market_2l_cola_revenue=market_revenue,
                full_soft_drinks_revenue=frame.revenue.sum(),
                market_coverage=selected_revenue / market_revenue,
                full_category_coverage=selected_revenue / frame.revenue.sum(),
            )
        )
    coverage = save_csv(rows, output, "revenue_coverage.csv")
    omissions = products.loc[
        products.market_2l_cola & ~products.upc.isin(selected.upc)
    ].copy()
    save_csv(
        omissions.sort_values(["split", "revenue"], ascending=[True, False]),
        output,
        "omitted_2l_cola_products.csv",
    )
    label = (
        "2L cola category"
        if coverage.market_coverage.min() >= settings["category_coverage_threshold"]
        else "four-product 2L cola portfolio"
    )
    save_json(
        dict(
            optimized_scope=label,
            coverage_threshold=settings["category_coverage_threshold"],
            excluded_ambiguous_descriptions=products.loc[products.ambiguous, "descrip"]
            .unique()
            .tolist(),
        ),
        output / "market_definition.json",
    )
