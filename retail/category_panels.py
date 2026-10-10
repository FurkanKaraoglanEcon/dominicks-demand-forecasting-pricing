"""Build auxiliary category portfolios from the same normalized movement.

Inputs: complete raw aggregates, normalized movement, cola selection and census.
Outputs: four-price category panels, training ranks, product mappings and coverage.
The cola research sample remains the primary portfolio. Cereal and detergent
are explicitly package portfolios with differing sizes, not whole categories.
"""

import logging
import numpy as np
import pandas as pd
from .io import save_csv
from .product_definitions import PRODUCTS, COLA_LABELS

LOG = logging.getLogger(__name__)


def select_other_portfolio(category, config):
    """Choose four observed package UPCs by revenue through the configured cutoff."""
    output = config["paths"]["output"]
    products = pd.read_csv(output / "categories" / f"{category}_products_by_split.csv")
    ranks = products[products.split.eq(1)].copy()
    # Coverage uses the study's training calendar, not future UPC survival.
    train_end = int(
        np.floor(
            config["data"]["first_week"]
            + (config["data"]["last_week"] - config["data"]["first_week"] + 1)
            * config["data"]["training_share"]
        )
        - 1
    )
    movements = pd.read_parquet(
        config["paths"]["processed"] / f"movement_{category}.parquet",
        columns=["upc", "store", "week"],
    )
    stores = movements[movements.week.le(train_end)].groupby("upc").store.nunique()
    ranks["training_stores"] = ranks.upc.map(stores)
    ranks["training_coverage"] = ranks.observations / (
        ranks.training_stores * train_end
    )
    ranks["eligible"] = (
        ranks.training_stores.ge(config["sample"]["minimum_stores"])
        & ranks.training_coverage.ge(config["sample"]["minimum_training_coverage"])
        & ranks["size"].str.contains(r"\d", na=False)
        & ranks.descrip.notna()
    )
    ranks = ranks.sort_values(["revenue", "upc"], ascending=[False, True])
    fixed_upcs = config["categories"][category].get("fixed_upcs", [])
    if fixed_upcs:
        chosen = ranks.set_index("upc").loc[fixed_upcs].reset_index()
        assert (
            chosen.eligible.all()
        ), "Declared benchmark UPCs must pass training eligibility"
    else:
        chosen = ranks[ranks.eligible].head(4).copy()
    assert (
        len(chosen) == 4
    ), f"Four sufficiently observed package UPCs required: {category}"
    ranks["selected"] = ranks.upc.isin(chosen.upc)
    save_csv(ranks, output / "tables", f"product_candidates_{category}.csv")
    chosen["category"], chosen["product"] = category, PRODUCTS
    chosen["label"] = chosen.descrip.str.strip()
    chosen["selection_last_week"] = train_end
    chosen["selection_rule"] = (
        "declared uploaded benchmark; training eligibility audited"
        if fixed_upcs
        else "training revenue among observed package UPCs; heterogeneous formats"
    )
    return chosen[
        [
            "category",
            "product",
            "upc",
            "label",
            "size",
            "selection_last_week",
            "selection_rule",
        ]
    ]


def common_price_panel(mapping, config):
    """Pivot actual four-product records; absent movement remains absent."""
    category = mapping.category.iloc[0]
    long = pd.read_parquet(
        config["paths"]["processed"] / f"movement_{category}.parquet",
        filters=[("upc", "in", mapping.upc.tolist())],
    )
    long = long.merge(mapping[["upc", "product"]], on="upc", validate="many_to_one")
    assert not long.duplicated(["store", "week", "product"]).any()
    fields = {"p": "price", "move": "units", "cost": "cost", "promo_flag": "promotion"}
    wide = long.pivot(index=["store", "week"], columns="product", values=list(fields))
    wide.columns = [f"{fields[field]}_{product}" for field, product in wide.columns]
    panel = wide.dropna(
        subset=[f"price_{p}" for p in PRODUCTS] + [f"units_{p}" for p in PRODUCTS]
    ).reset_index()
    return panel


def merge_auxiliary_data(panel, category, config):
    """Attach observed weekly totals, seven-day traffic and log census income."""
    processed = config["paths"]["processed"]
    demo = pd.read_parquet(processed / "store_demographics.parquet")
    traffic = pd.read_parquet(processed / "weekly_customers.parquet")
    totals = pd.read_parquet(processed / f"category_weekly_{category}.parquet")
    panel = panel.merge(demo, on="store", how="left", validate="many_to_one")
    panel["income_centered"] = (
        panel.income - panel[["store", "income"]].drop_duplicates("store").income.mean()
    )
    panel = panel.merge(
        traffic[["store", "week", "customers"]],
        on=["store", "week"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(totals, on=["store", "week"], how="left", validate="one_to_one")
    panel["portfolio_units"] = panel[[f"units_{p}" for p in PRODUCTS]].sum(axis=1)
    panel["portfolio_revenue"] = sum(
        panel[f"price_{p}"] * panel[f"units_{p}"] for p in PRODUCTS
    )
    assert (panel.portfolio_revenue <= panel.category_revenue + 1e-6).all()
    if category == "sdr":
        metadata = pd.read_csv(processed / "metadata_sdr.csv")
        mask = metadata["size"].str.match(config["data"]["size_pattern"], na=False)
        segment = pd.read_parquet(
            processed / "movement_sdr.parquet",
            columns=["store", "week", "move", "revenue"],
            filters=[("upc", "in", metadata.loc[mask, "upc"].tolist())],
        )
        segment = segment.groupby(["store", "week"], as_index=False).agg(
            two_liter_units=("move", "sum"), two_liter_revenue=("revenue", "sum")
        )
        panel = panel.merge(
            segment, on=["store", "week"], how="left", validate="one_to_one"
        )
    panel = panel.sort_values(["store", "week"]).reset_index(drop=True)
    panel.to_parquet(processed / f"panel_{category}.parquet", index=False)
    return panel


def prepare_category_panels(cola, selected, config):
    """Execute category construction once and export visible mappings and coverage."""
    rename = {f"move{j}": f"units_p{j}" for j in range(1, 5)}
    rename.update({f"p{j}": f"price_p{j}" for j in range(1, 5)})
    rename.update({f"cost{j}": f"cost_p{j}" for j in range(1, 5)})
    rename.update({f"promo_flag{j}": f"promotion_p{j}" for j in range(1, 5)})
    keep = ["store", "week"] + list(rename)
    cola = cola[keep].rename(columns=rename)
    mapping = selected[["upc", "size", "product"]].copy()
    mapping["product"] = PRODUCTS
    mapping["category"] = "sdr"
    mapping["label"] = mapping["product"].map(COLA_LABELS)
    mapping["selection_last_week"] = int(
        np.floor(
            config["data"]["first_week"]
            + (config["data"]["last_week"] - config["data"]["first_week"] + 1)
            * config["data"]["training_share"]
        )
        - 1
    )
    mapping["selection_rule"] = "training-ranked comparable plain regular/diet 2L cola"
    mappings = [mapping]
    panels = {"sdr": merge_auxiliary_data(cola, "sdr", config)}
    for category in ["cer", "lnd"]:
        mapping = select_other_portfolio(category, config)
        panels[category] = merge_auxiliary_data(
            common_price_panel(mapping, config), category, config
        )
        mappings.append(mapping)
    save_csv(
        pd.concat(mappings, ignore_index=True),
        config["paths"]["output"] / "tables",
        "product_mapping.csv",
    )
    coverage = []
    for category, panel in panels.items():
        totals = pd.read_parquet(
            config["paths"]["processed"] / f"category_weekly_{category}.parquet"
        )
        coverage.append(
            dict(
                category=category,
                markets=len(panel),
                stores=panel.store.nunique(),
                first_week=panel.week.min(),
                last_week=panel.week.max(),
                portfolio_share_complete_markets=panel.portfolio_revenue.sum()
                / panel.category_revenue.sum(),
                whole_category_markets=len(totals),
                scope="four-product portfolio",
            )
        )
        LOG.info(
            "%s auxiliary panel: %s markets, %s stores, weeks %s-%s",
            category,
            len(panel),
            panel.store.nunique(),
            panel.week.min(),
            panel.week.max(),
        )
    save_csv(coverage, config["paths"]["output"] / "tables", "sample_coverage.csv")
