"""Read every raw row once; retain narrow-market detail and complete category totals."""

import hashlib
import logging
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .io import save_csv, save_json, save_panel, calendar_lag

LOG = logging.getLogger(__name__)


def read_metadata(path, settings, category):
    data = pd.read_csv(path, encoding="cp1252", dtype={"UPC": "int64"})
    data.columns = data.columns.str.lower()
    for name in ["descrip", "size"]:
        data[name] = data[name].fillna("").str.strip().str.upper()
    assert not data.upc.duplicated().any(), "UPC metadata must be unique"
    if category == "sdr":
        size_ok = data["size"].str.match(settings["size_pattern"])
        data["ambiguous"] = size_ok & data.descrip.str.contains(
            settings["ambiguous_pattern"]
        )
        data["market_2l_cola"] = (
            size_ok
            & data.descrip.str.contains(settings["cola_pattern"])
            & ~data.ambiguous
        )
        data["candidate"] = (
            data.market_2l_cola
            & data.descrip.str.contains(settings["brand_pattern"])
            & ~data.descrip.str.contains(settings["variant_pattern"])
        )
        data["brand"] = np.select(
            [
                data.descrip.str.contains("PEPSI"),
                data.descrip.str.contains("COKE|COCA"),
            ],
            ["Pepsi", "Coke"],
            default="Other",
        )
        data["diet"] = data.descrip.str.contains("DIET").astype(int)
    return data


def movement_chunks(path, chunk_rows):
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"One complete CSV expected: {path}")
        with archive.open(members[0]) as stream:
            reader = pd.read_csv(
                stream,
                encoding="cp1252",
                chunksize=chunk_rows,
                float_precision="round_trip",
                dtype={"PRICE_HEX": str, "PROFIT_HEX": str},
            )
            for chunk in reader:
                chunk.columns = chunk.columns.str.lower()
                for name in ["price", "profit"]:
                    values = chunk[name + "_hex"].astype(str)
                    assert values.str.fullmatch(
                        r"[0-9a-fA-F]{16}"
                    ).all(), "Malformed hexadecimal source value"
                    decoded = np.frombuffer(
                        bytes.fromhex("".join(values)), dtype=">f8"
                    ).astype(float)
                    assert np.allclose(
                        chunk[name], decoded, atol=1e-8, rtol=1e-8, equal_nan=True
                    )
                    chunk[name] = decoded
                yield chunk.drop(columns=["price_hex", "profit_hex"])


def valid_movement(data, settings):
    required = ["store", "upc", "week", "move", "qty", "price"]
    return (
        data[required].notna().all(axis=1)
        & data.ok.eq(1)
        & data.store.gt(0)
        & data.store.eq(np.floor(data.store))
        & data.week.eq(np.floor(data.week))
        & data.week.between(settings["first_week"], settings["last_week"])
        & data.move.ge(0)
        & data.qty.gt(0)
        & data.price.gt(0)
    )


def split_code(week, train_end, validation_end):
    return np.where(week <= train_end, 1, np.where(week <= validation_end, 2, 3))


def audit_category(raw, category, settings, cutoffs, output):
    metadata = read_metadata(raw / f"upc{category}.csv", settings, category)
    totals, markets, key_hashes, retained = [], [], [], []
    counts = dict(
        category=category,
        raw_rows=0,
        invalid_rows=0,
        zero_sales_rows=0,
        unknown_metadata_rows=0,
        invalid_cost_rows=0,
        market_raw_rows=0,
    )
    for chunk in movement_chunks(raw / f"w{category}.zip", settings["chunk_rows"]):
        counts["raw_rows"] += len(chunk)
        counts["unknown_metadata_rows"] += int((~chunk.upc.isin(metadata.upc)).sum())
        valid = valid_movement(chunk, settings)
        counts["invalid_rows"] += int((~valid).sum())
        if category == "sdr":
            broad = chunk.upc.isin(
                metadata.loc[metadata.market_2l_cola | metadata.ambiguous, "upc"]
            )
            counts["market_raw_rows"] += int(broad.sum())
        clean = chunk.loc[valid].copy()
        key_hashes.append(
            pd.util.hash_pandas_object(
                clean[["store", "upc", "week"]], index=False
            ).to_numpy()
        )
        counts["zero_sales_rows"] += int(clean.move.eq(0).sum())
        clean["p"] = clean.price / clean.qty
        clean["revenue"] = clean.p * clean.move
        clean["margin"] = clean.profit / 100
        cost_ok = clean.margin.notna() & clean.margin.ge(0) & clean.margin.lt(1)
        counts["invalid_cost_rows"] += int((~cost_ok).sum())
        # Invalid margins remove only accounting calculations, not demand observations.
        clean["cost"] = (clean.p * (1 - clean.margin)).where(cost_ok)
        clean["grossprofit"] = (clean.revenue * clean.margin).where(cost_ok)
        clean["promo_flag"] = (
            clean.sale.fillna("")
            .str.strip()
            .str.upper()
            .isin(["B", "C", "S"])
            .astype(int)
        )
        clean["split"] = split_code(clean.week, *cutoffs)
        totals.append(
            clean.groupby(["upc", "split"], as_index=False).agg(
                observations=("move", "size"),
                units=("move", "sum"),
                revenue=("revenue", "sum"),
                grossprofit=("grossprofit", "sum"),
                cost_observations=("cost", "count"),
                promotion_rows=("promo_flag", "sum"),
            )
        )
        markets.append(
            clean.groupby(["store", "week"], as_index=False).agg(
                revenue=("revenue", "sum"),
                observations=("move", "size"),
                grossprofit=("grossprofit", "sum"),
            )
        )
        if category == "sdr":
            retained.append(clean.loc[broad.loc[valid]])
    hashes = np.concatenate(key_hashes)
    assert len(np.unique(hashes)) == len(hashes), f"Duplicate valid keys in {category}"
    # A hash collision would fail this conservative check, never silently drop a row.
    products = (
        pd.concat(totals)
        .groupby(["upc", "split"], as_index=False)
        .sum()
        .merge(metadata, on="upc", how="left", validate="many_to_one")
    )
    weekly = pd.concat(markets).groupby(["store", "week"], as_index=False).sum()
    counts.update(
        valid_rows=len(hashes),
        products=products.upc.nunique(),
        stores=weekly.store.nunique(),
        store_weeks=len(weekly),
        revenue=weekly.revenue.sum(),
        duplicate_valid_keys=0,
    )
    save_csv(products, output, f"{category}_products_by_split.csv")
    save_csv(weekly, output, f"{category}_store_week.csv")
    leaders = (
        products.loc[products.split.eq(1)]
        .sort_values(["revenue", "upc"], ascending=[False, True])
        .head(10)
        .copy()
    )
    leaders["training_revenue_share"] = (
        leaders.revenue / products.loc[products.split.eq(1), "revenue"].sum()
    )
    save_csv(leaders, output, f"{category}_training_leaders.csv")
    LOG.info(
        "%s: read %s rows; %s valid; %s metadata mismatches",
        category,
        counts["raw_rows"],
        counts["valid_rows"],
        counts["unknown_metadata_rows"],
    )
    detail = pd.concat(retained, ignore_index=True) if retained else None
    return metadata, products, weekly, counts, detail


def select_products(movement, metadata, settings, train_end):
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


def weekly_traffic(path, settings, output):
    traffic = pd.read_stata(
        path,
        convert_categoricals=False,
        columns=["store", "date", "week", "custcoun", "grocery"],
    )
    raw_rows = len(traffic)
    traffic["day"] = pd.to_datetime(traffic.date, errors="coerce", format="mixed")
    expected_week = np.floor((traffic.day - pd.Timestamp("1989-09-14")).dt.days / 7) + 1
    usable = (
        traffic.store.gt(0)
        & traffic.store.eq(np.floor(traffic.store))
        & traffic.week.between(settings["first_week"], settings["last_week"])
        & traffic.day.notna()
        & traffic.week.eq(expected_week)
    )
    traffic = traffic.loc[usable].drop_duplicates()
    assert not traffic.duplicated(
        ["store", "day"]
    ).any(), "Conflicting daily customer counts"
    traffic.loc[traffic.custcoun.le(0), "custcoun"] = np.nan
    weekly = traffic.groupby(["store", "week"], as_index=False).agg(
        traffic=("custcoun", "sum"),
        grocery=("grocery", "sum"),
        traffic_days=("custcoun", "count"),
        grocery_days=("grocery", "count"),
    )
    weekly.loc[weekly.traffic_days.ne(7), "traffic"] = np.nan
    weekly.loc[weekly.grocery_days.ne(7), "grocery"] = np.nan
    save_json(
        dict(
            raw_rows=raw_rows,
            valid_daily_rows=len(traffic),
            weekly_rows=len(weekly),
            complete_traffic_weeks=int(weekly.traffic.notna().sum()),
            household_ids_available=False,
        ),
        output / "customer_count_audit.json",
    )
    save_csv(weekly, output, "weekly_traffic.csv")
    return weekly


def omitted_index(movement, selected, train_end, settings, output):
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


def add_covariates(panel, raw, category_weekly, traffic):
    demo = pd.read_stata(raw / "demo.dta", convert_categoricals=False)
    demo = demo.loc[
        demo.store.notna() & demo.store.gt(0),
        [
            "store",
            "zone",
            "scluster",
            "income",
            "poverty",
            "hsizeavg",
            "nocar",
            "educ",
            "age60",
            "ethnic",
        ],
    ]
    assert not demo.store.duplicated().any()
    panel = panel.merge(demo, on="store", how="left", validate="many_to_one")
    panel["income_c"] = (
        panel.income - panel[["store", "income"]].drop_duplicates("store").income.mean()
    )
    panel = panel.merge(
        traffic, on=["store", "week"], how="left", validate="one_to_one"
    )
    for category in ["cer", "lnd"]:
        weekly = category_weekly[category][["store", "week", "revenue"]].rename(
            columns={"revenue": f"revenue_{category}"}
        )
        panel = panel.merge(
            weekly, on=["store", "week"], how="left", validate="one_to_one"
        )
        panel[f"lag_activity_{category}"] = calendar_lag(
            panel, f"revenue_{category}", 1
        )
        panel[f"lag_activity_{category}"] = np.log1p(panel[f"lag_activity_{category}"])
    panel["lag_log_traffic"] = np.log(calendar_lag(panel, "traffic", 1))
    panel["segment_units"] = panel[[f"move{j}" for j in range(1, 5)]].sum(axis=1)
    panel["segment_revenue"] = sum(
        panel[f"move{j}"] * panel[f"p{j}"] for j in range(1, 5)
    )
    panel["units_per_visit"] = panel.segment_units / panel.traffic
    panel["start_date"] = pd.Timestamp("1989-09-14") + pd.to_timedelta(
        7 * (panel.week - 1), unit="D"
    )
    panel["trend"] = (panel.week - 1) / 52
    for harmonic in [1, 2]:
        angle = 2 * np.pi * harmonic * (panel.week - 1) / 52.1775
        panel[f"sin{harmonic}"] = np.sin(angle)
        panel[f"cos{harmonic}"] = np.cos(angle)
    for j in range(1, 5):
        panel[f"lag1_{j}"] = calendar_lag(panel, f"zq{j}", 1)
        panel[f"lag52_{j}"] = calendar_lag(panel, f"move{j}", 52)
        panel[f"lag52z_{j}"] = calendar_lag(panel, f"zq{j}", 52)
    return panel


def report_coverage(products, selected, output, settings):
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


def prepare(config):
    settings, paths = config["data"], config["paths"]
    raw, output = paths["raw"], paths["output"]
    span = settings["last_week"] - settings["first_week"] + 1
    train_end = int(
        np.floor(settings["first_week"] + span * settings["training_share"]) - 1
    )
    validation_end = int(
        np.floor(
            settings["first_week"]
            + span * (settings["training_share"] + settings["validation_share"])
        )
        - 1
    )
    cutoffs = (train_end, validation_end)
    summaries, weekly = [], {}
    for category in settings["categories"]:
        meta, products, markets, counts, retained = audit_category(
            raw, category, settings, cutoffs, output / "categories"
        )
        summaries.append(counts)
        weekly[category] = markets
        if category == "sdr":
            metadata, sdr_products, movement = meta, products, retained
    save_csv(summaries, output / "categories", "category_summary.csv")
    hashes = []
    for path in sorted(raw.iterdir()):
        if path.is_file() and path.suffix in [".zip", ".csv", ".dta", ".pdf"]:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            hashes.append(
                dict(file=path.name, bytes=path.stat().st_size, sha256=digest)
            )
    save_json(hashes, output / "raw_checksums.json")
    movement = movement.merge(
        metadata[["upc", "ambiguous"]], on="upc", validate="many_to_one"
    )
    selected, candidates = select_products(movement, metadata, settings, train_end)
    save_csv(selected, output, "selected_products.csv")
    save_csv(candidates, output, "candidate_training_coverage.csv")
    save_csv(
        metadata.loc[metadata.market_2l_cola | metadata.ambiguous],
        output,
        "eligible_upcs.csv",
    )
    report_coverage(sdr_products, selected, output, settings)
    panel, audit = build_panel(movement, selected, cutoffs, settings, output)
    index = omitted_index(movement, selected, train_end, settings, output)
    panel = panel.merge(index, on=["store", "week"], how="left", validate="one_to_one")
    traffic = weekly_traffic(raw / "ccount.dta", settings, output)
    panel = add_covariates(panel, raw, weekly, traffic)
    weights = []
    for family, pair in enumerate([(1, 2), (3, 4)], 1):
        revenues = np.array(
            [
                (
                    panel.loc[panel.split.eq(1), f"move{j}"]
                    * panel.loc[panel.split.eq(1), f"p{j}"]
                ).sum()
                for j in pair
            ]
        )
        shares = revenues / revenues.sum()
        for j, weight in zip(pair, shares):
            weights.append(dict(family=family, product=j, weight=weight))
        panel[f"fq{family}"] = panel[[f"move{j}" for j in pair]].sum(axis=1)
        panel[f"flq{family}"] = np.log(
            panel[f"fq{family}"].where(panel[f"fq{family}"].gt(0))
        )
        panel[f"flp{family}"] = sum(
            weight * panel[f"lp{j}"] for j, weight in zip(pair, shares)
        )
        panel[f"wedge{family}"] = panel[f"lp{pair[0]}"] - panel[f"lp{pair[1]}"]
    save_csv(weights, output, "family_weights.csv")
    for name in (
        [f"lp{j}" for j in range(1, 5)]
        + ["flp1", "flp2"]
        + [f"promo_flag{j}" for j in range(1, 5)]
    ):
        for lag in range(1, config["demand"]["dynamic_lags"] + 1):
            panel[f"{name}_lag{lag}"] = calendar_lag(panel, name, lag)
    audit.update(
        broad_valid_rows=len(movement),
        demographics_stores=panel.loc[panel.income.notna(), "store"].nunique(),
        traffic_markets=int(panel.traffic.notna().sum()),
        input_price_precision="decoded source float64",
    )
    save_panel(panel, output / "analysis_panel.dta")
    save_json(audit, output / "preparation_audit.json")
    LOG.info(
        "Selected %s; panel %s markets, %s stores; cutoffs %s",
        selected.upc.tolist(),
        len(panel),
        panel.store.nunique(),
        cutoffs,
    )
    return panel, selected
