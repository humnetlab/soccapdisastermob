#!/usr/bin/env python3
"""
03_processing_lbs.py -- processing of location-based service smartphone traces.

Produces the supplementary figures characterising the mobility panel:

    figure_s4.png  User selection: time span against number of stay points.
    figure_s5.png  Departure time, stay duration and location rank.
    figure_s6.png  Census against smartphone population, and expansion factors.
    figure_s7.png  Evacuee counts by nighttime threshold, and distances.

A user counts as an evacuee if the ZIP code where they spent most of their
nighttime hours during the disaster week differs from the one where they spent
most of their nighttime hours over the four preceding weeks. Nighttime runs
20:00-07:00 and the main analysis requires 480 minutes of dwell in a week.

Figures S5 and S6 report Michael and Florence jointly, as they share one 2018
processing run; figure S7 separates them, since evacuation is defined relative
to a single landfall date.

The raw traces cannot be redistributed, so the script runs in two stages:
``--stage figures`` (default) draws from the shared aggregates in ``data/`` and
needs no raw data; ``--stage aggregate`` rebuilds those aggregates from the
proprietary panel. Run with --help for options.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Non-interactive backend: the script only ever writes files.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


# --------------------------------------------------------------------------
# Study configuration
# --------------------------------------------------------------------------

#: Disaster groupings used for the mobility and population figures (S5, S6).
#: Michael and Florence share a single 2018 processing run and are reported
#: jointly, following the published figures.
PANEL_DISASTERS = {
    "Harvey": {
        "dataset": "20170701_20171231_tx_combined",
        "state_fips": ["48"],
        "bq_tables": ["census_tracts_texas"],
    },
    "Irma": {
        "dataset": "20170701_20171231_fl_combined",
        "state_fips": ["12"],
        "bq_tables": ["census_tracts_florida"],
    },
    "Imelda": {
        "dataset": "20190701_20191231_full_combined",
        "state_fips": ["48"],
        "bq_tables": ["census_tracts_texas"],
    },
    "Michael & Florence": {
        "dataset": "20180701_20181231_full_combined",
        "state_fips": ["12", "01", "47", "37", "13", "45"],
        "bq_tables": [
            "census_tracts_florida",
            "census_tracts_alabama",
            "census_tracts_tennessee",
            "census_tracts_north_carolina",
            "census_tracts_georgia",
            "census_tracts_south_carolina",
        ],
    },
}

#: Individual storms used for the evacuation figure (S7). Michael and Florence
#: are separated here because evacuation is defined relative to a single
#: landfall date.
EVAC_DISASTERS = [
    {"key": "harvey", "label": "Harvey (2017)",
     "dataset": "20170701_20171231_tx_combined", "event_date": date(2017, 8, 25)},
    {"key": "irma", "label": "Irma (2017)",
     "dataset": "20170701_20171231_fl_combined", "event_date": date(2017, 9, 10)},
    {"key": "florence", "label": "Florence (2018)",
     "dataset": "20180701_20181231_full_combined", "event_date": date(2018, 9, 14)},
    {"key": "michael", "label": "Michael (2018)",
     "dataset": "20180701_20181231_full_combined", "event_date": date(2018, 10, 10)},
    {"key": "imelda", "label": "Imelda (2019)",
     "dataset": "20190701_20191231_full_combined", "event_date": date(2019, 9, 17)},
]

#: Dataset used for the user-selection figure (S4).
S4_DATASET = "20170701_20171231_fl_combined"

#: Stay-point / active-day bounds defining a high-quality user.
NUM_STAY_POINTS_RANGE = [100, 800]
TIME_SPAN_DAYS_RANGE = [15, 30]

#: Nighttime dwell thresholds (minutes) probed for the evacuation definition.
THRESHOLDS = [240, 480, 720, 960, 1200, 1440, 1680, 1920]
MAIN_THRESHOLD = 480
PRE_WEEKS = 4
POST_WEEKS = 8
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 7
MIN_CANDIDATE_H3_MINUTES = 60

#: Parquet sub-directories within a processed dataset.
LABELED_STAYS = "StayPointsLabeled_home_work_other"
HOME_WORK_STAYS = "StayPointsWithHomeWork"

#: Colour palette shared by the evacuation panels.
EVAC_PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2"]

#: Palette used by the ``sparkmobility`` user-selection plot, reproduced here so
#: that the figure stage does not require the package to be installed.
USER_SELECTION_COLORS = [
    "#c45161", "#e094a0", "#f2b6c0", "#f2dde1",
    "#cbc7d8", "#8db7d2", "#5e62a9", "#434279",
]

#: Aggregate table filenames written by ``--stage aggregate`` and read by
#: ``--stage figures``.
AGG_FILES = {
    "s4": "s4_user_activity_levels.csv",
    "s5": "s5_mobility_distributions.csv",
    "s6": "s6_census_vs_smartphone.csv",
    "s7_thresholds": "s7_threshold_summary.csv",
    "s7_distance": "s7_evacuation_distance_kde.csv",
}

#: Optional per-user evacuation cohort, written only when
#: ``--write-evacuation-cohort`` is given. This is an intermediate of the
#: figure S7 pipeline that later scripts reuse; it is a per-person record and
#: is therefore not part of the shared data.
EVACUATION_COHORT_FILE = "evacuation_cohort.csv"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def log(message: str) -> None:
    """Print a progress message immediately."""
    print(message, flush=True)


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """Fail early with a readable message if an aggregate table is malformed."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(df.columns)}"
        )


def read_aggregate(agg_dir: Path, key: str) -> pd.DataFrame:
    """Load one aggregate table, with a message pointing at the other stage."""
    path = agg_dir / AGG_FILES[key]
    if not path.exists():
        raise SystemExit(
            f"Missing aggregate table: {path}\n"
            f"Run `python {Path(__file__).name} --stage aggregate` with access to "
            f"the raw LBS panel to regenerate it, or restore the file shipped "
            f"with the repository."
        )
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Stage 1: aggregation from the raw LBS panel
# --------------------------------------------------------------------------

def build_s4_aggregate(spark, dataset_dir: Path) -> pd.DataFrame:
    """User activity levels: distinct users per (time span, stay-point count).

    This mirrors the aggregation inside
    ``sparkmobility.processing.user_selection.UserSelection.filter_users``:
    stay points are grouped per user to obtain a count and an observation span,
    and users are then counted per (span, count) pair. The result contains no
    identifiers.
    """
    from pyspark.sql import functions as F

    df = spark.read.parquet(str(dataset_dir / HOME_WORK_STAYS))

    df_grouped = (
        df.groupBy("caid")
        .agg(
            F.count("*").alias("num_stays"),
            F.min("stay_start_timestamp").alias("first_stay"),
            F.max("stay_end_timestamp").alias("last_stay"),
        )
        .withColumn(
            "duration_days",
            F.round(
                (F.unix_timestamp("last_stay") - F.unix_timestamp("first_stay")) / 86400
            ).cast("int"),
        )
    )

    active_level = (
        df_grouped.groupBy("duration_days", "num_stays")
        .count()
        .orderBy("duration_days", "num_stays")
        .toPandas()
    )

    total_users = df.select("caid").distinct().count()
    selected = df_grouped.filter(
        (F.col("num_stays") >= NUM_STAY_POINTS_RANGE[0])
        & (F.col("num_stays") <= NUM_STAY_POINTS_RANGE[1])
        & (F.col("duration_days") >= TIME_SPAN_DAYS_RANGE[0])
        & (F.col("duration_days") <= TIME_SPAN_DAYS_RANGE[1])
    )
    log(f"  Total users: {total_users:,}; high-quality users: {selected.count():,}")

    return active_level


def build_s5_aggregate(df, label: str) -> pd.DataFrame:
    """Departure-time, stay-duration and location-rank distributions.

    ``df`` is the labeled stay-point DataFrame. Returns a long table with
    columns ``disaster``, ``quantity``, ``x``, ``y``; all three quantities are
    population-level distributions.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    frames = []

    # Departure time: share of stays beginning in each hour of the day.
    dep_rows = (
        df.groupBy("hour_of_day")
        .agg(F.count("*").alias("count"))
        .orderBy("hour_of_day")
        .collect()
    )
    x_t = np.array([r["hour_of_day"] for r in dep_rows], dtype=float)
    pdf_t = np.array([r["count"] for r in dep_rows], dtype=float)
    pdf_t /= pdf_t.sum()
    frames.append(pd.DataFrame({"quantity": "departure_time", "x": x_t, "y": pdf_t}))

    # Stay duration: 0.5 h bins over 0-24 h.
    dur_rows = (
        df.withColumn(
            "stay_duration_hours",
            (
                F.col("stay_end_timestamp").cast("double")
                - F.col("stay_start_timestamp").cast("double")
            ) / 3600.0,
        )
        .filter((F.col("stay_duration_hours") > 0) & (F.col("stay_duration_hours") <= 24))
        .withColumn("bin_idx", F.floor(F.col("stay_duration_hours") * 2.0))
        .groupBy("bin_idx")
        .agg(F.count("*").alias("count"))
        .orderBy("bin_idx")
        .collect()
    )
    bin_idx = np.array([r["bin_idx"] for r in dur_rows], dtype=float)
    count_dur = np.array([r["count"] for r in dur_rows], dtype=float)
    frames.append(pd.DataFrame({
        "quantity": "stay_duration",
        "x": (bin_idx + 0.5) / 2.0,
        "y": count_dur / count_dur.sum(),
    }))

    # Location rank: mean visitation frequency of a user's L-th ranked location.
    user_loc = df.groupBy("caid", "h3_id_region").agg(F.count("*").alias("visits"))
    w_user = Window.partitionBy("caid")
    w_rank = Window.partitionBy("caid").orderBy(F.desc("visits"))
    rank_rows = (
        user_loc.withColumn("total_visits", F.sum("visits").over(w_user))
        .withColumn("freq", F.col("visits") / F.col("total_visits"))
        .withColumn("rank", F.dense_rank().over(w_rank))
        .groupBy("rank")
        .agg(F.avg("freq").alias("avg_freq"))
        .orderBy("rank")
        .collect()
    )
    frames.append(pd.DataFrame({
        "quantity": "location_rank",
        "x": np.array([r["rank"] for r in rank_rows], dtype=float),
        "y": np.array([r["avg_freq"] for r in rank_rows], dtype=float),
    }))

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "disaster", label)
    return out


def build_s6_aggregate(
    df, label: str, state_fips: list[str],
    bq_tables: list[str], census_api_key: str,
) -> pd.DataFrame:
    """Census population vs. smartphone-estimated population, per census tract.

    ``df`` is the labeled stay-point DataFrame. Home locations are counted per
    H3 cell, each cell is assigned to the census tract with the nearest
    internal point, and the resulting counts are joined to ACS 5-year tract
    populations.
    """
    import h3
    from census import Census
    from google.cloud import bigquery
    from pyspark.sql import functions as F

    home_counts = (
        df.filter(F.col("home_h3_index").isNotNull())
        .select("caid", "home_h3_index")
        .dropDuplicates(["caid", "home_h3_index"])
        .groupBy("home_h3_index")
        .agg(F.countDistinct("caid").alias("mobile_count"))
        .withColumnRenamed("home_h3_index", "h3_index")
        .toPandas()
    )
    home_counts["avg_lat"] = home_counts["h3_index"].apply(lambda h: h3.cell_to_latlng(h)[0])
    home_counts["avg_lon"] = home_counts["h3_index"].apply(lambda h: h3.cell_to_latlng(h)[1])

    census = Census(census_api_key)
    results = []
    for state in state_fips:
        rows = census.acs5.state_county_tract(
            fields=("B01003_001E",), state_fips=state,
            county_fips="*", tract="*", year=2019,
        )
        results.append(pd.DataFrame(rows))
    census_df = pd.concat(results, ignore_index=True).rename(
        columns={"B01003_001E": "population"}
    )
    census_df["population"] = census_df["population"].astype(int)
    census_df["geoid"] = census_df["state"] + census_df["county"] + census_df["tract"]

    parts = [
        "SELECT geo_id AS geoid, ST_Y(internal_point_geo) AS lat, "
        f"ST_X(internal_point_geo) AS lon "
        f"FROM `bigquery-public-data.geo_census_tracts.{table}`"
        for table in bq_tables
    ]
    sql = "WITH tracts AS (" + " UNION ALL ".join(parts) + ") SELECT * FROM tracts"
    tracts_df = bigquery.Client().query(sql).to_dataframe()

    # Assign each H3 cell to the census tract with the nearest internal point.
    # A KD-tree keeps this linear-ish in the number of cells; the brute-force
    # fallback matches it exactly but scales as cells x tracts.
    tract_coords = np.column_stack([
        tracts_df["lat"].to_numpy(dtype=float),
        tracts_df["lon"].to_numpy(dtype=float),
    ])
    tract_geoid = tracts_df["geoid"].to_numpy()
    cell_coords = np.column_stack([
        home_counts["avg_lat"].to_numpy(dtype=float),
        home_counts["avg_lon"].to_numpy(dtype=float),
    ])

    try:
        from scipy.spatial import cKDTree

        _, nearest_idx = cKDTree(tract_coords).query(cell_coords, k=1)
    except ImportError:
        nearest_idx = np.array([
            np.argmin(((tract_coords - point) ** 2).sum(axis=1))
            for point in cell_coords
        ])

    home_counts["geoid"] = tract_geoid[nearest_idx]
    mobile_df = home_counts.groupby("geoid", as_index=False)["mobile_count"].sum()

    merged = census_df.merge(mobile_df, on="geoid", how="left")
    merged["mobile_count"] = merged["mobile_count"].fillna(0)

    # Only the two population columns are needed downstream; drop the GEOID so
    # the shared table cannot be re-linked to specific tracts.
    return pd.DataFrame({
        "disaster": label,
        "population": merged["population"].astype(int),
        "mobile_count": merged["mobile_count"].astype(int),
    })


def _night_window_sql(parquet_glob: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """SQL selecting nighttime stays, anchored to the night they belong to."""
    query_start = (pd.Timestamp(start) - pd.Timedelta(days=1)).date()
    query_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).date()
    return f"""
        WITH night_rows AS (
            SELECT
                CAST(caid AS VARCHAR) AS pid,
                CAST(h3_index AS VARCHAR) AS stay_h3,
                CASE
                    WHEN hour_of_day < {NIGHT_END_HOUR}
                    THEN CAST(local_time AS DATE) - INTERVAL 1 DAY
                    ELSE CAST(local_time AS DATE)
                END AS night_anchor_date,
                CAST(stay_duration AS DOUBLE) / 60000000.0 AS night_minutes
            FROM read_parquet('{parquet_glob}')
            WHERE CAST(local_time AS DATE) >= DATE '{query_start}'
              AND CAST(local_time AS DATE) <  DATE '{query_end}'
              AND stay_duration > 0
              AND h3_index IS NOT NULL
              AND (hour_of_day >= {NIGHT_START_HOUR} OR hour_of_day < {NIGHT_END_HOUR})
        )
        SELECT pid,
               CAST(date_trunc('week', night_anchor_date) AS DATE) AS week_start,
               stay_h3,
               SUM(night_minutes) AS night_minutes
        FROM night_rows
        GROUP BY 1, 2, 3
    """


def get_windows(event_date: date) -> dict[str, pd.Timestamp]:
    """Pre-disaster, disaster and post-disaster week boundaries."""
    week_start = pd.Timestamp(event_date) - pd.Timedelta(
        days=pd.Timestamp(event_date).weekday()
    )
    return {
        "pre_start": week_start - pd.Timedelta(weeks=PRE_WEEKS),
        "disaster_week_start": week_start,
        "post_start": week_start + pd.Timedelta(weeks=1),
        "post_end": week_start + pd.Timedelta(weeks=1 + POST_WEEKS),
    }


def build_s7_aggregates(
    lbs_base_dir: Path, zcta_shapefile: Path, duckdb_threads: int,
    collect_cohort: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evacuation counts across thresholds, and evacuation-distance densities.

    A user is an evacuee if their dominant nighttime ZIP during the disaster
    week differs from their dominant nighttime ZIP over the preceding weeks.
    Returns ``(threshold_summary, distance_kde, cohort)``. Only the smoothed
    distance density is published -- per-evacuee distances are never written to
    disk. ``cohort`` holds the per-user evacuation status at the main threshold
    and is empty unless ``collect_cohort`` is set; it is an intermediate that
    later scripts reuse rather than part of the shared data.
    """
    import duckdb
    import geopandas as gpd
    import h3
    from scipy.stats import gaussian_kde
    from shapely.geometry import box

    # ZIP code tabulation areas, used to convert H3 cells into a stable
    # geography for the "same place / different place" comparison.
    raw = gpd.read_file(zcta_shapefile)
    zip_col = next(
        c for c in ["ZCTA5CE20", "ZCTA5CE10", "ZCTA5CE"] if c in raw.columns
    )
    zcta = raw[[zip_col, "geometry"]].rename(columns={zip_col: "zip"})
    zcta["zip"] = zcta["zip"].astype(str).str.strip().str.zfill(5)
    if zcta.crs is None:
        zcta = zcta.set_crs("EPSG:4326", allow_override=True)
    zcta = zcta.dropna(subset=["geometry"]).drop_duplicates("zip").reset_index(drop=True)

    zcta_3857 = zcta.to_crs("EPSG:3857")
    centroids = zcta_3857.geometry.centroid
    zip_coords = pd.DataFrame({
        "zip": zcta_3857["zip"], "x": centroids.x, "y": centroids.y,
    }).drop_duplicates("zip")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={duckdb_threads}")

    threshold_parts: list[pd.DataFrame] = []
    distance_parts: list[pd.DataFrame] = []
    cohort_parts: list[pd.DataFrame] = []
    x_grid = np.linspace(0, 200, 500)

    try:
        for cfg in EVAC_DISASTERS:
            log(f"  {cfg['label']} ...")
            windows = get_windows(cfg["event_date"])
            glob_path = str(lbs_base_dir / cfg["dataset"] / HOME_WORK_STAYS / "*.parquet")

            weekly = con.execute(
                _night_window_sql(glob_path, windows["pre_start"], windows["post_end"])
            ).df()
            if weekly.empty:
                log("    no nighttime stays found; skipping")
                continue

            weekly["week_start"] = pd.to_datetime(weekly["week_start"])
            weekly = weekly.loc[weekly["night_minutes"] >= MIN_CANDIDATE_H3_MINUTES]
            if weekly.empty:
                log("    no candidate cells above the minimum dwell; skipping")
                continue

            # Map each candidate H3 cell to a ZIP, restricted to the storm's
            # bounding box so the spatial join stays cheap.
            cells = sorted(weekly["stay_h3"].dropna().astype(str).unique().tolist())
            latlon = np.array([h3.cell_to_latlng(c) for c in cells], dtype=float)
            bbox = box(
                latlon[:, 1].min() - 0.75, latlon[:, 0].min() - 0.75,
                latlon[:, 1].max() + 0.75, latlon[:, 0].max() + 0.75,
            )
            local = zcta[zcta.intersects(bbox)]
            if local.empty:
                local = zcta

            points = gpd.GeoDataFrame(
                {"stay_h3": cells},
                geometry=gpd.points_from_xy(latlon[:, 1], latlon[:, 0]),
                crs=local.crs,
            )
            joined = gpd.sjoin(points, local, how="left", predicate="within")
            lookup = (
                joined[["stay_h3", "zip"]]
                .sort_values(["stay_h3", "zip"], na_position="last")
                .drop_duplicates("stay_h3", keep="first")
            )

            weekly_zip = (
                weekly.merge(lookup, on="stay_h3", how="left")
                .dropna(subset=["zip"])
                .groupby(["pid", "week_start", "zip"], as_index=False)["night_minutes"]
                .sum()
            )
            weekly_zip = weekly_zip.loc[weekly_zip["night_minutes"] >= min(THRESHOLDS)]
            if weekly_zip.empty:
                log("    no ZIP-weeks above the minimum threshold; skipping")
                continue

            pre_start = windows["pre_start"]
            week_start = windows["disaster_week_start"]
            week_end = week_start + pd.Timedelta(weeks=1)

            for threshold in THRESHOLDS:
                eligible = weekly_zip.loc[weekly_zip["night_minutes"] >= threshold]
                # One dominant ZIP per user-week.
                weekly_top = (
                    eligible.sort_values(
                        ["pid", "week_start", "night_minutes", "zip"],
                        ascending=[True, True, False, True],
                    )
                    .drop_duplicates(["pid", "week_start"], keep="first")
                )

                def dominant(frame, start, end, column):
                    window = frame.loc[
                        (frame["week_start"] >= start) & (frame["week_start"] < end)
                    ]
                    if window.empty:
                        return pd.DataFrame(columns=["pid", column])
                    return (
                        window.groupby(["pid", "zip"], as_index=False)["night_minutes"]
                        .sum()
                        .sort_values(
                            ["pid", "night_minutes", "zip"],
                            ascending=[True, False, True],
                        )
                        .drop_duplicates("pid", keep="first")
                        .rename(columns={"zip": column})[["pid", column]]
                    )

                pre_home = dominant(weekly_top, pre_start, week_start, "pre_zip")
                crisis = dominant(weekly_top, week_start, week_end, "crisis_zip")
                merged = pre_home.merge(crisis, on="pid", how="left")
                merged["evacuated"] = (
                    merged["crisis_zip"].notna()
                    & (merged["crisis_zip"] != merged["pre_zip"])
                ).astype(int)

                threshold_parts.append(pd.DataFrame([{
                    "disaster": cfg["label"],
                    "threshold_minutes": threshold,
                    "n_users_with_pre_home": int(pre_home["pid"].nunique()),
                    "n_users_with_disaster_week_location": int(crisis["pid"].nunique()),
                    "n_evacuated": int(merged["evacuated"].sum()),
                    "evacuation_rate": float(merged["evacuated"].mean())
                    if len(merged) else np.nan,
                }]))

                if threshold == MAIN_THRESHOLD:
                    if collect_cohort:
                        cohort = merged[["pid", "pre_zip", "crisis_zip", "evacuated"]].copy()
                        cohort.insert(1, "disaster", cfg["label"])
                        cohort_parts.append(cohort)

                    # Straight-line distance between ZIP centroids, evaluated as
                    # a smoothed density so no per-person record is published.
                    evacuees = merged.loc[merged["evacuated"] == 1]
                    dist = (
                        evacuees
                        .merge(zip_coords.add_prefix("o_"), left_on="pre_zip",
                               right_on="o_zip", how="left")
                        .merge(zip_coords.add_prefix("d_"), left_on="crisis_zip",
                               right_on="d_zip", how="left")
                    )
                    km = np.sqrt(
                        (dist["d_x"] - dist["o_x"]) ** 2
                        + (dist["d_y"] - dist["o_y"]) ** 2
                    ) / 1000.0
                    km = km.dropna()
                    km = km[km <= x_grid.max()].astype(float)
                    if len(km) >= 5 and km.nunique() >= 2:
                        density = gaussian_kde(km)(x_grid)
                        density[x_grid <= 1] = 0.0
                        distance_parts.append(pd.DataFrame({
                            "disaster": cfg["label"],
                            "distance_km": x_grid,
                            "density": density,
                        }))
                    log(f"    {int(merged['evacuated'].sum()):,} evacuees at "
                        f"{MAIN_THRESHOLD} min")
    finally:
        con.close()

    threshold_summary = (
        pd.concat(threshold_parts, ignore_index=True)
        if threshold_parts else pd.DataFrame()
    )
    distance_kde = (
        pd.concat(distance_parts, ignore_index=True)
        if distance_parts else pd.DataFrame()
    )
    cohort = (
        pd.concat(cohort_parts, ignore_index=True)
        if cohort_parts else pd.DataFrame()
    )
    return threshold_summary, distance_kde, cohort


def create_spark_session(cores: int, memory_gb: int, temp_dir: str | None):
    """Build a local Spark session for the aggregation queries.

    The aggregations below use only standard DataFrame operations, so no
    custom JAR is required and the session is constructed directly rather than
    through ``sparkmobility``.
    """
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName("soccapdisastermob-lbs")
        .config("spark.executor.memory", f"{memory_gb}g")
        .config("spark.driver.memory", f"{memory_gb}g")
        .config("spark.sql.files.ignoreCorruptFiles", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1000")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
    )
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        builder = (
            builder.config("spark.local.dir", temp_dir)
            .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={temp_dir}")
            .config("spark.executor.extraJavaOptions", f"-Djava.io.tmpdir={temp_dir}")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def run_aggregate_stage(args: argparse.Namespace) -> None:
    """Regenerate every aggregate table from the proprietary LBS panel."""
    lbs_base = Path(args.lbs_base_dir).expanduser().resolve()
    if not lbs_base.is_dir():
        raise SystemExit(f"--lbs-base-dir does not exist: {lbs_base}")

    agg_dir = Path(args.agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)

    targets = set(args.figures)

    spark = None
    if targets & {"s4", "s5", "s6"}:
        spark = create_spark_session(args.cores, args.memory_gb, args.spark_temp_dir)

    try:
        if "s4" in targets:
            log("Aggregating user activity levels (figure S4) ...")
            out = build_s4_aggregate(spark, lbs_base / S4_DATASET)
            out.to_csv(agg_dir / AGG_FILES["s4"], index=False)
            log(f"  wrote {agg_dir / AGG_FILES['s4']}")

        # Figures S5 and S6 both read the labeled stay points, so when both are
        # requested the dataset is cached once and reused across the two sets
        # of aggregations rather than being scanned twice.
        if targets & {"s5", "s6"}:
            if "s6" in targets and not args.census_api_key:
                raise SystemExit(
                    "--census-api-key is required to rebuild the figure S6 aggregate."
                )

            s5_parts, s6_parts = [], []
            for label, cfg in PANEL_DISASTERS.items():
                log(f"Processing labeled stay points for {label} ...")
                labeled = spark.read.parquet(
                    str(lbs_base / cfg["dataset"] / LABELED_STAYS)
                )
                labeled.persist()
                try:
                    if "s5" in targets:
                        log("  mobility distributions (figure S5) ...")
                        s5_parts.append(build_s5_aggregate(labeled, label))
                    if "s6" in targets:
                        log("  census vs. smartphone population (figure S6) ...")
                        s6_parts.append(build_s6_aggregate(
                            labeled, label, cfg["state_fips"], cfg["bq_tables"],
                            args.census_api_key,
                        ))
                finally:
                    labeled.unpersist()

            if s5_parts:
                pd.concat(s5_parts, ignore_index=True).to_csv(
                    agg_dir / AGG_FILES["s5"], index=False
                )
                log(f"  wrote {agg_dir / AGG_FILES['s5']}")
            if s6_parts:
                pd.concat(s6_parts, ignore_index=True).to_csv(
                    agg_dir / AGG_FILES["s6"], index=False
                )
                log(f"  wrote {agg_dir / AGG_FILES['s6']}")
    finally:
        if spark is not None:
            spark.stop()

    if "s7" in targets:
        if not args.zcta_shapefile:
            raise SystemExit(
                "--zcta-shapefile is required to rebuild the figure S7 aggregate."
            )
        log("Aggregating evacuation statistics (figure S7) ...")
        threshold_summary, distance_kde, cohort = build_s7_aggregates(
            lbs_base, Path(args.zcta_shapefile).expanduser().resolve(),
            args.duckdb_threads, collect_cohort=bool(args.write_evacuation_cohort),
        )
        threshold_summary.to_csv(agg_dir / AGG_FILES["s7_thresholds"], index=False)
        distance_kde.to_csv(agg_dir / AGG_FILES["s7_distance"], index=False)
        log(f"  wrote {agg_dir / AGG_FILES['s7_thresholds']}")
        log(f"  wrote {agg_dir / AGG_FILES['s7_distance']}")

        if args.write_evacuation_cohort and not cohort.empty:
            cohort_path = Path(args.write_evacuation_cohort)
            cohort_path.parent.mkdir(parents=True, exist_ok=True)
            cohort.to_csv(cohort_path, index=False)
            log(f"  wrote {cohort_path} ({len(cohort):,} users)")


# --------------------------------------------------------------------------
# Stage 2: figures from the shared aggregates
# --------------------------------------------------------------------------

def _visualize_user_selection(active_level, num_stay_points_range, time_span_days_range):
    """Fallback copy of ``UserSelection._visualize`` for environments without
    ``sparkmobility`` installed. Kept byte-for-byte equivalent in output."""
    pivot = active_level.pivot(
        index="duration_days", columns="num_stays", values="count"
    ).fillna(0)
    log_data = np.log10(pivot + 1)
    masked_data = np.ma.masked_where(pivot.values == 0, log_data.values)

    cmap = LinearSegmentedColormap.from_list("custom_cmap_masked", USER_SELECTION_COLORS)
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.grid(False)
    ax.minorticks_off()
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    im = ax.imshow(masked_data, aspect="auto", cmap=cmap)

    ax.add_patch(plt.Rectangle(
        (num_stay_points_range[0], time_span_days_range[0]),
        num_stay_points_range[1] - num_stay_points_range[0],
        time_span_days_range[1] - time_span_days_range[0],
        linewidth=2, edgecolor=USER_SELECTION_COLORS[-1], facecolor="none",
    ))

    num_cols = log_data.shape[1]
    xticks = (np.linspace(0, num_cols - 1, 10, dtype=int)
              if num_cols > 10 else np.arange(num_cols))
    ax.set_xticks(xticks)
    ax.set_xticklabels(log_data.columns[xticks], rotation=90)

    num_rows = log_data.shape[0]
    yticks = (np.linspace(0, num_rows - 1, 10, dtype=int)
              if num_rows > 10 else np.arange(num_rows))
    ax.set_yticks(yticks)
    ax.set_yticklabels(log_data.index[yticks])
    ax.invert_yaxis()

    ax.set_xlabel("Number of Stay Points")
    ax.set_ylabel("Time Span (Days)")
    fig.colorbar(im, ax=ax).set_label("Log10 Number of Users")
    plt.tight_layout()
    return fig, ax


def plot_figure_s4(agg_dir: Path, plots_dir: Path, dpi: int) -> None:
    """Figure S4: user selection heatmap over time span and stay-point count."""
    active_level = read_aggregate(agg_dir, "s4")
    require_columns(active_level, ["duration_days", "num_stays", "count"],
                    AGG_FILES["s4"])

    # Prefer the published implementation so the figure matches the package
    # exactly; fall back to the vendored copy when it is not installed.
    try:
        from sparkmobility.processing.user_selection import UserSelection
        import seaborn as sns

        sns.set(style="whitegrid", font_scale=1.5)
        fig, _ = UserSelection._visualize(
            active_level, NUM_STAY_POINTS_RANGE, TIME_SPAN_DAYS_RANGE
        )
    except ImportError:
        log("  sparkmobility not installed; using the bundled plotting routine")
        fig, _ = _visualize_user_selection(
            active_level, NUM_STAY_POINTS_RANGE, TIME_SPAN_DAYS_RANGE
        )

    out = plots_dir / "figure_s4.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def plot_figure_s5(agg_dir: Path, plots_dir: Path, dpi: int) -> None:
    """Figure S5: departure time, stay duration and location rank per disaster."""
    data = read_aggregate(agg_dir, "s5")
    require_columns(data, ["disaster", "quantity", "x", "y"], AGG_FILES["s5"])

    disasters = [d for d in PANEL_DISASTERS if d in set(data["disaster"])]
    columns = [
        ("departure_time", "Departure Time", "Hour of day", "tab:blue"),
        ("stay_duration", "Stay Duration", "Duration [hours]", "tab:red"),
        ("location_rank", "Location Rank", "Rank $L$", "tab:green"),
    ]

    fig, axes = plt.subplots(
        len(disasters), 3, figsize=(16, 4 * len(disasters)), dpi=dpi, squeeze=False
    )
    last_row = len(disasters) - 1

    for row, disaster in enumerate(disasters):
        for col, (quantity, title, xlabel, color) in enumerate(columns):
            ax = axes[row, col]
            sub = data[
                (data["disaster"] == disaster) & (data["quantity"] == quantity)
            ].sort_values("x")
            ax.plot(sub["x"], sub["y"], "o-", color=color, linewidth=2, markersize=3)

            if quantity == "departure_time":
                ax.set_xlim(-0.5, 23.5)
                ax.set_xticks([0, 6, 12, 18, 24])
            elif quantity == "stay_duration":
                ax.set_yscale("log")
                ax.set_xlim(0, 24)
                ax.set_xticks([0, 12, 24])
            else:
                ax.set_xscale("log")
                ax.set_yscale("log")

            ax.tick_params(labelsize=20)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            if row == 0:
                ax.set_title(title, fontsize=20)
            if row == last_row:
                ax.set_xlabel(xlabel, fontsize=20)
            if col == 0:
                ax.set_ylabel(disaster, fontsize=20)

    plt.tight_layout()
    out = plots_dir / "figure_s5.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def plot_figure_s6(agg_dir: Path, plots_dir: Path, dpi: int) -> None:
    """Figure S6: census vs. smartphone population, and expansion factors."""
    data = read_aggregate(agg_dir, "s6")
    require_columns(data, ["disaster", "population", "mobile_count"], AGG_FILES["s6"])

    disasters = [d for d in PANEL_DISASTERS if d in set(data["disaster"])]
    n = len(disasters)

    bins = np.logspace(1, 4, 50)
    ef_bins = np.linspace(0, 300, 50)
    shades = ["#eeeeee", "#bbbbbb", "#888888", "#444444", "#000000"]

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), dpi=dpi, squeeze=False)

    for col, disaster in enumerate(disasters):
        sub = data[data["disaster"] == disaster]

        # Top row: joint distribution of census and smartphone population.
        ax = axes[0, col]
        counts, xedges, yedges = np.histogram2d(
            sub["population"].values, sub["mobile_count"].values, bins=[bins, bins]
        )
        if np.any(counts > 0):
            boundaries = np.quantile(counts[counts > 0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
            # Quantiles collapse when the histogram is nearly uniform; fall back
            # to a linear scale so BoundaryNorm receives increasing edges.
            if len(np.unique(boundaries)) == len(boundaries):
                norm = BoundaryNorm(boundaries, ncolors=len(shades))
            else:
                norm = BoundaryNorm(
                    np.linspace(counts[counts > 0].min(), counts.max(), len(shades) + 1),
                    ncolors=len(shades),
                )
            ax.pcolormesh(
                xedges, yedges, counts.T,
                cmap=ListedColormap(shades), norm=norm, shading="auto",
            )

        ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], ls=":", color="black", lw=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([10, 100, 10000])
        ax.set_yticks([10, 100, 10000])
        ax.tick_params(labelsize=20)
        ax.set_title(disaster, fontsize=20)
        if col == 0:
            ax.set_ylabel("Smartphone\nEstimated Population", fontsize=20)

        # Bottom row: distribution of the census / smartphone expansion factor.
        ax = axes[1, col]
        expansion = (
            (sub["population"] / sub["mobile_count"])
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        ax.hist(expansion, bins=ef_bins, alpha=0.7, color="black",
                edgecolor="black", linewidth=0.5)
        ax.grid(True, which="both", ls="-", alpha=0.3, color="grey")
        ax.set_xticks([0, 150, 300])
        ax.set_xlim(0, 300)
        ax.tick_params(labelsize=20)
        if col == 0:
            ax.set_ylabel("Number of Units", fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)

    row0_y = min(axes[0, col].get_position().y0 for col in range(n))
    fig.text(0.5, row0_y - 0.1, "Census Population", ha="center", fontsize=20)
    fig.text(0.5, -0.02, "Expansion Factor (Census / Smartphone)",
             ha="center", fontsize=20)

    out = plots_dir / "figure_s6.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def plot_figure_s7(agg_dir: Path, plots_dir: Path, dpi: int) -> None:
    """Figure S7: evacuee counts by nighttime threshold, and travel distances."""
    thresholds = read_aggregate(agg_dir, "s7_thresholds")
    distances = read_aggregate(agg_dir, "s7_distance")
    require_columns(thresholds, ["disaster", "threshold_minutes", "n_evacuated"],
                    AGG_FILES["s7_thresholds"])
    require_columns(distances, ["disaster", "distance_km", "density"],
                    AGG_FILES["s7_distance"])

    order = [cfg["label"] for cfg in EVAC_DISASTERS]
    present = [d for d in order if d in set(thresholds["disaster"])]
    colors = {d: EVAC_PALETTE[i % len(EVAC_PALETTE)] for i, d in enumerate(present)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 5), dpi=dpi)

    # Left: sensitivity of the evacuee count to the nighttime dwell threshold.
    ax = axes[0]
    for disaster in present:
        sub = thresholds[thresholds["disaster"] == disaster].sort_values(
            "threshold_minutes"
        )
        ax.plot(sub["threshold_minutes"], sub["n_evacuated"], marker="o",
                linewidth=2, markersize=5, color=colors[disaster], label=disaster)
    ax.axvline(MAIN_THRESHOLD, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Nighttime threshold (minutes)", fontsize=20)
    ax.set_ylabel("Number of evacuees", fontsize=20)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.tick_params(labelsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))

    # Right: distribution of evacuation distances at the main threshold.
    ax = axes[1]
    for disaster in present:
        sub = distances[distances["disaster"] == disaster].sort_values("distance_km")
        if sub.empty:
            continue
        ax.plot(sub["distance_km"], sub["density"], color=colors[disaster],
                linewidth=2.2, label=disaster)
    ax.set_xlabel("Evacuation distance (km)", fontsize=20)
    ax.set_ylabel("Density", fontsize=20)
    ax.set_xlim(0, 200)
    ax.tick_params(labelsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.legend(frameon=False, fontsize=20)

    out = plots_dir / "figure_s7.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def run_figures_stage(args: argparse.Namespace) -> None:
    """Render the requested figures from the shared aggregate tables."""
    agg_dir = Path(args.agg_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    renderers = {
        "s4": plot_figure_s4,
        "s5": plot_figure_s5,
        "s6": plot_figure_s6,
        "s7": plot_figure_s7,
    }
    for key in args.figures:
        log(f"Rendering figure {key.upper()} ...")
        renderers[key](agg_dir, plots_dir, args.dpi)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage", choices=["figures", "aggregate"], default="figures",
        help="'figures' (default) renders plots from the shared aggregates; "
             "'aggregate' rebuilds those aggregates from the raw LBS panel.",
    )
    parser.add_argument(
        "--figures", nargs="+", choices=["s4", "s5", "s6", "s7"],
        default=["s4", "s5", "s6", "s7"],
        help="Subset of figures to produce (default: all).",
    )
    parser.add_argument(
        "--data-dir", default=str(here / "data"),
        help="Repository data directory (default: %(default)s).",
    )
    parser.add_argument(
        "--agg-dir", default=None,
        help="Directory holding the LBS aggregate tables "
             "(default: <data-dir>/processed/lbs_aggregates).",
    )
    parser.add_argument(
        "--plots-dir", default=str(here / "plots"),
        help="Directory for the output figures (default: %(default)s).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Figure resolution (default: %(default)s).",
    )

    raw = parser.add_argument_group("raw data options (--stage aggregate only)")
    raw.add_argument(
        "--lbs-base-dir",
        help="Directory containing the processed stay-point datasets. "
             "The raw LBS data is proprietary and is not distributed with "
             "this repository.",
    )
    raw.add_argument(
        "--zcta-shapefile",
        help="Path to the TIGER/Line ZCTA shapefile (required for figure S7).",
    )
    raw.add_argument(
        "--census-api-key",
        help="Your own US Census API key, required for figure S6. Request one "
             "free of charge at https://api.census.gov/data/key_signup.html.",
    )
    raw.add_argument(
        "--duckdb-threads", type=int, default=8,
        help="Threads used by DuckDB (default: %(default)s).",
    )
    raw.add_argument(
        "--cores", type=int, default=8,
        help="Cores for the local Spark session (default: %(default)s).",
    )
    raw.add_argument(
        "--memory-gb", type=int, default=32,
        help="Driver/executor memory in GB (default: %(default)s).",
    )
    raw.add_argument(
        "--spark-temp-dir", default=None,
        help="Scratch directory for Spark shuffle files.",
    )
    raw.add_argument(
        "--write-evacuation-cohort", metavar="PATH", default=None,
        help="Also write the per-user evacuation status at the main threshold "
             "to PATH, for reuse by later scripts. This is a per-person record "
             "and must not be committed to the repository; write it outside "
             "the data directory.",
    )

    args = parser.parse_args(argv)
    if args.agg_dir is None:
        args.agg_dir = str(Path(args.data_dir) / "processed" / "lbs_aggregates")

    if args.stage == "aggregate" and not args.lbs_base_dir:
        parser.error("--lbs-base-dir is required when --stage aggregate")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "aggregate":
        run_aggregate_stage(args)
    else:
        run_figures_stage(args)
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
