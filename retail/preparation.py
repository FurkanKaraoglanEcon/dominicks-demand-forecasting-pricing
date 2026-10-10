"""Read every raw row once; retain narrow-market detail and complete category totals.

Inputs: original scanner ZIPs, UPC metadata, demographics, visits and configuration.
Outputs: normalized movement, category totals, selected cola panel and preparation audits.
"""

import hashlib
import logging
import zipfile
import numpy as np
import pandas as pd
from .io import save_csv, save_json, save_panel, calendar_lag
from .io import split_code
from .selection import select_products, omitted_index, build_panel, report_coverage
import pyarrow as pa
import pyarrow.parquet as pq

LOG = logging.getLogger(__name__)


def read_metadata(path, settings, category):
    """Return unique UPC metadata and explicit description/size-based cola flags. Settings hold the visible market definitions."""
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
    """Yield every CSV chunk from the zipped source after verifying and decoding price/profit precision."""
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
    """Return the valid priced-row mask using source flags, calendar keys and nonnegative movement; missing price is not zero demand."""
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


def audit_category(raw, category, settings, cutoffs, output):
    """Stream one complete category, save normalized rows and training/category totals, and return metadata, aggregates, raw counts and narrow-market detail."""
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
    processed = output.parent / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    movement_path = processed / f"movement_{category}.parquet"
    writer = None
    volume_totals = []
    for chunk in movement_chunks(raw / f"w{category}.zip", settings["chunk_rows"]):
        counts["raw_rows"] += len(chunk)
        counts["unknown_metadata_rows"] += int((~chunk.upc.isin(metadata.upc)).sum())
        valid_units = (
            chunk.ok.eq(1)
            & chunk.move.ge(0)
            & chunk.qty.gt(0)
            & chunk.store.gt(0)
            & chunk.store.eq(np.floor(chunk.store))
            & chunk.week.eq(np.floor(chunk.week))
            & chunk.week.between(settings["first_week"], settings["last_week"])
        )
        volume = chunk.loc[
            valid_units, ["store", "week", "upc", "move", "price", "qty"]
        ].copy()
        priced = volume.price.gt(0) & np.isfinite(volume.price)
        volume["category_revenue"] = (volume.price / volume.qty * volume.move).where(
            priced, 0
        )
        volume["priced_units"] = volume.move.where(priced, 0)
        volume["active_upcs"] = (priced & volume.move.gt(0)).astype(int)
        volume_totals.append(
            volume.groupby(["store", "week"], as_index=False).agg(
                category_units=("move", "sum"),
                category_revenue=("category_revenue", "sum"),
                priced_units=("priced_units", "sum"),
                active_upcs=("active_upcs", "sum"),
            )
        )
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
        # A loss leader can have negative gross margin and valid positive AAC.
        # Only nonfinite or nonpositive acquisition cost is unusable.
        cost_ok = np.isfinite(clean.margin) & clean.margin.lt(1)
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
        normalized = clean[
            [
                "store",
                "upc",
                "week",
                "move",
                "p",
                "cost",
                "margin",
                "promo_flag",
                "revenue",
                "grossprofit",
            ]
        ].copy()
        normalized[["store", "upc", "week"]] = normalized[
            ["store", "upc", "week"]
        ].astype("int64")
        normalized["promo_flag"] = normalized.promo_flag.astype("int8")
        arrow_table = pa.Table.from_pandas(normalized, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                movement_path, arrow_table.schema, compression="zstd"
            )
        writer.write_table(arrow_table)
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
    if writer is not None:
        writer.close()
    category_totals = (
        pd.concat(volume_totals).groupby(["store", "week"], as_index=False).sum()
    )
    category_totals.to_parquet(
        processed / f"category_weekly_{category}.parquet", index=False
    )
    metadata.to_csv(processed / f"metadata_{category}.csv", index=False)
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


def weekly_traffic(path, settings, output):
    """Return seven-day complete positive weekly visit totals; nonnegative daily counts are valid and incomplete weeks remain missing."""
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
    traffic.loc[traffic.custcoun.lt(0), "custcoun"] = np.nan
    weekly = traffic.groupby(["store", "week"], as_index=False).agg(
        traffic=("custcoun", "sum"),
        grocery=("grocery", "sum"),
        traffic_days=("custcoun", "count"),
        grocery_days=("grocery", "count"),
    )
    weekly.loc[weekly.traffic_days.ne(7) | weekly.traffic.le(0), "traffic"] = np.nan
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
    weekly.rename(columns={"traffic": "customers"}).to_parquet(
        output / "processed" / "weekly_customers.parquet", index=False
    )
    return weekly


def add_covariates(panel, raw, category_weekly, traffic):
    """Return the panel merged with census/visits and calendar-dated activity and quantity lags."""
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
    # The supplied income variable is already log median census income.
    demo.to_parquet(
        category_weekly["processed"] / "store_demographics.parquet", index=False
    )
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


def prepare(config):
    """Read all configured raw inputs, select the training-ranked cola portfolio and return the prepared research panel and UPC mapping."""
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
    weekly["processed"] = output / "processed"
    panel = add_covariates(panel, raw, weekly, traffic)
    weekly_summary = panel.groupby(["week", "start_date"], as_index=False).agg(
        segment_units=("segment_units", "sum"),
        segment_revenue=("segment_revenue", "sum"),
        units_per_visit=("units_per_visit", "mean"),
    )
    save_csv(weekly_summary, output, "weekly_segment_summary.csv")
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
    panel.to_parquet(output / "processed" / "cola_research_panel.parquet", index=False)
    save_json(audit, output / "preparation_audit.json")
    LOG.info(
        "Selected %s; panel %s markets, %s stores; cutoffs %s",
        selected.upc.tolist(),
        len(panel),
        panel.store.nunique(),
        cutoffs,
    )
    return panel, selected
