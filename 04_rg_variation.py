#!/usr/bin/env python3
"""
04_rg_variation.py -- variation in radius of gyration around landfall.

    figure_3.png   Mean radius of gyration over days [-7, +7] per storm,
                   stratified by wind intensity, each paired with a map of the
                   intensity across residential ZIP codes, plus the evacuation
                   rate against intensity.
    table_s2.csv   Post-landfall change in radius of gyration, per bin (Eq. 2).
    table_s3.csv   Pooled estimates with intensity rank as an ordered
                   covariate (Eq. 3).

For each user and day the radius of gyration is the duration-weighted root mean
squared distance of the ZIP codes they visited from their duration-weighted
centre of mass, on ZIP centroids in Web Mercator. Daily values are averaged
within disaster, day, intensity bin and income group, and smoothed with a
three-day centred mean before both plotting and fitting.

Intensity comes from HURDAT2, binned by maximum sustained wind at each
residential ZIP code: <34, 34-49, 50-63 and >=64 knots. Imelda appears in the
figure but not the regressions: no ZIP code in its footprint reached
tropical-storm force, so the intensity gradient is not identified for it.

The raw traces cannot be redistributed, so the script runs in two stages:
``--stage figures`` (default) draws from the shared aggregates in ``data/`` and
needs no raw data; ``--stage aggregate`` rebuilds them. Run with --help.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


# --------------------------------------------------------------------------
# Study configuration
# --------------------------------------------------------------------------

#: Days relative to landfall retained in the panel.
T_LO, T_HI = -7, 7

#: Landfall dates and the processed dataset each storm was drawn from.
DISASTERS = [
    {"key": "florence", "label": "Florence (2018)",
     "dataset": "20180701_20181231_full_combined", "event_date": date(2018, 9, 14)},
    {"key": "irma", "label": "Irma (2017)",
     "dataset": "20170701_20171231_fl_combined", "event_date": date(2017, 9, 10)},
    {"key": "imelda", "label": "Imelda (2019)",
     "dataset": "20190701_20191231_full_combined", "event_date": date(2019, 9, 17)},
    {"key": "harvey", "label": "Harvey (2017)",
     "dataset": "20170701_20171231_tx_combined", "event_date": date(2017, 8, 25)},
    {"key": "michael", "label": "Michael (2018)",
     "dataset": "20180701_20181231_full_combined", "event_date": date(2018, 10, 10)},
]

#: Panel layout of figure 3: two rows of three, the last cell holding the
#: evacuation-rate panels rather than a disaster.
FIGURE_ORDER = [
    "Florence (2018)", "Irma (2017)", "Imelda (2019)",
    "Harvey (2017)", "Michael (2018)",
]

#: Storms entering the regressions. Imelda is excluded: every residential ZIP
#: code fell below tropical-storm force, so the intensity gradient in equations
#: (2) and (3) is not identified for it.
REGRESSION_DISASTERS = [
    "Florence (2018)", "Harvey (2017)", "Irma (2017)", "Michael (2018)",
]

#: Wind intensity bins, in increasing order. The rank of a bin in this list is
#: the ordered covariate k in equation (3).
INTENSITY_ORDER = [
    "No TS winds (<34kt)",
    "Tropical-storm (34–49kt)",
    "Gale-force (50–63kt)",
    "Hurricane-force (>=64kt)",
]
INTENSITY_COLORS = {
    "No TS winds (<34kt)": "#7f7f7f",
    "Tropical-storm (34–49kt)": "#2ca02c",
    "Gale-force (50–63kt)": "#ff7f0e",
    "Hurricane-force (>=64kt)": "#d62728",
}
INTENSITY_SHORT = {
    "No TS winds (<34kt)": "No TS winds",
    "Tropical-storm (34–49kt)": "Tropical-storm",
    "Gale-force (50–63kt)": "Gale-force",
    "Hurricane-force (>=64kt)": "Hurricane-force",
}

#: Minimum dwell in an H3 cell, in minutes, for that cell to enter the radius
#: of gyration. Matches the threshold used in the evacuation pipeline.
MIN_CANDIDATE_H3_MINUTES = 60

#: Nighttime dwell threshold defining the evacuation cohort, in minutes.
MAIN_THRESHOLD = 480

#: Parquet sub-directory holding the labeled stay points.
HOME_WORK_STAYS = "StayPointsWithHomeWork"

#: Aggregate tables written by ``--stage aggregate``.
AGG_FILES = {
    "panel": "rg_intensity_panel.csv",
    "zip_intensity": "zip_intensity.csv",
    "evac_rate": "evacuation_rate_by_intensity.csv",
}

#: ZIP code boundaries for the map panels. Only the ZIP codes inside the five
#: storm footprints are shipped, simplified to roughly 500 m, which keeps the
#: file small enough to distribute while remaining faithful at figure scale.
FOOTPRINT_GEOMETRY = "zcta_storm_footprints.gpkg"

FS = 20


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def log(message: str) -> None:
    print(message, flush=True)


def norm_zip(values) -> pd.Series:
    """Normalise ZIP codes to zero-padded five-character strings."""
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(5)
    )


def intensity_rank(label: str) -> int:
    """Ordered rank k of an intensity bin; 99 for anything unrecognised."""
    return INTENSITY_ORDER.index(label) if label in INTENSITY_ORDER else 99


def lighten(hex_color: str, factor: float = 0.52) -> tuple[float, float, float]:
    """Blend a colour toward white, used for the low-income series."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (1 - (1 - r) * (1 - factor),
            1 - (1 - g) * (1 - factor),
            1 - (1 - b) * (1 - factor))


def read_aggregate(agg_dir: Path, key: str, required: bool = True) -> pd.DataFrame:
    """Load one aggregate table, pointing at the other stage if it is absent."""
    path = agg_dir / AGG_FILES[key]
    if not path.exists():
        if not required:
            return pd.DataFrame()
        raise SystemExit(
            f"Missing aggregate table: {path}\n"
            f"Run `python {Path(__file__).name} --stage aggregate` with access "
            f"to the raw LBS panel to regenerate it, or restore the file "
            f"shipped with the repository."
        )
    return pd.read_csv(path, dtype={"home_zip": str})


def smooth_panel(panel: pd.DataFrame, value_col: str = "mean_rog") -> pd.DataFrame:
    """Three-day centred rolling mean within each series of the panel."""
    parts = []
    for _, group in panel.groupby(["disaster", "int_bin", "income_group"]):
        group = group.sort_values("t").copy()
        group[value_col] = (
            group[value_col].rolling(window=3, center=True, min_periods=1).mean()
        )
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Stage 1: aggregation from the raw LBS panel
# --------------------------------------------------------------------------

def build_zip_centroids(zcta_shapefile: Path) -> dict[str, tuple[float, float]]:
    """Representative interior point of every ZCTA, in Web Mercator metres."""
    import geopandas as gpd

    zcta = gpd.read_file(zcta_shapefile)
    zip_col = next(
        (c for c in ["ZCTA5CE20", "ZCTA5CE10", "ZCTA5CE"] if c in zcta.columns),
        next(c for c in zcta.columns if "ZCTA5" in c.upper()),
    )
    zcta = zcta[[zip_col, "geometry"]].rename(columns={zip_col: "zip"})
    zcta["zip"] = norm_zip(zcta["zip"])
    zcta = zcta.dropna(subset=["geometry", "zip"]).drop_duplicates("zip")
    zcta = zcta.to_crs("EPSG:3857")

    # A representative point is guaranteed to lie inside the polygon, unlike a
    # centroid, which matters for the coastal ZCTAs in this sample.
    points = zcta.geometry.representative_point()
    return dict(zip(zcta["zip"].to_numpy(), zip(points.x.to_numpy(), points.y.to_numpy())))


def compute_daily_rg(
    con, dataset_dir: Path, event_date: date,
    coords: dict[str, tuple[float, float]], h3_to_zip: dict[str, str | None],
) -> pd.DataFrame:
    """Daily radius of gyration per user, in kilometres.

    Stays are aggregated to (user, day, ZIP) dwell time, a duration-weighted
    centre of mass is taken over the ZIPs visited that day, and the radius of
    gyration is the duration-weighted RMS distance from that centre.
    """
    day_lo = (pd.Timestamp(event_date) + pd.Timedelta(days=T_LO)).date()
    day_hi = (pd.Timestamp(event_date) + pd.Timedelta(days=T_HI)).date()
    glob_path = str(dataset_dir / HOME_WORK_STAYS / "*.parquet")

    daily = con.execute(f"""
        SELECT CAST(caid AS VARCHAR) AS pid,
               CAST(local_time AS DATE) AS stay_date,
               CAST(h3_index AS VARCHAR) AS stay_h3,
               SUM(CAST(stay_duration AS DOUBLE) / 60000000.0) AS stay_minutes
        FROM read_parquet('{glob_path}')
        WHERE CAST(local_time AS DATE) >= DATE '{day_lo}'
          AND CAST(local_time AS DATE) <= DATE '{day_hi}'
          AND stay_duration > 0
          AND h3_index IS NOT NULL
        GROUP BY 1, 2, 3
    """).df()
    if daily.empty:
        return pd.DataFrame(columns=["pid", "t", "rg_km"])

    # Discard brief stays before the expensive H3 -> ZIP mapping.
    daily = daily.loc[daily["stay_minutes"] >= MIN_CANDIDATE_H3_MINUTES]
    if daily.empty:
        return pd.DataFrame(columns=["pid", "t", "rg_km"])

    daily["zip"] = daily["stay_h3"].map(h3_to_zip)
    daily = daily.dropna(subset=["zip"])
    daily = daily.groupby(["pid", "stay_date", "zip"], as_index=False)["stay_minutes"].sum()

    xy = daily["zip"].map(lambda z: coords.get(z, (np.nan, np.nan)))
    daily["x"] = [p[0] for p in xy]
    daily["y"] = [p[1] for p in xy]
    daily = daily.dropna(subset=["x", "y"])
    if daily.empty:
        return pd.DataFrame(columns=["pid", "t", "rg_km"])

    total = daily.groupby(["pid", "stay_date"])["stay_minutes"].transform("sum")
    daily["w"] = daily["stay_minutes"] / total

    centre = daily.assign(wx=daily["w"] * daily["x"], wy=daily["w"] * daily["y"]) \
        .groupby(["pid", "stay_date"], as_index=False) \
        .agg(x_cm=("wx", "sum"), y_cm=("wy", "sum"))

    merged = daily.merge(centre, on=["pid", "stay_date"], how="inner")
    merged["w_dist_sq"] = merged["w"] * (
        (merged["x"] - merged["x_cm"]) ** 2 + (merged["y"] - merged["y_cm"]) ** 2
    )

    rg = merged.groupby(["pid", "stay_date"], as_index=False).agg(
        rg_sq=("w_dist_sq", "sum")
    )
    rg["rg_km"] = np.sqrt(rg["rg_sq"]) / 1000.0
    rg["t"] = (pd.to_datetime(rg["stay_date"]) - pd.Timestamp(event_date)).dt.days
    return rg[["pid", "t", "rg_km"]]


def map_h3_to_zip(cells: list[str], zcta_shapefile: Path) -> dict[str, str | None]:
    """Assign each H3 cell to the ZCTA containing its centre."""
    import geopandas as gpd
    import h3
    from shapely.geometry import box

    if not cells:
        return {}

    latlon = np.array([h3.cell_to_latlng(c) for c in cells], dtype=float)

    zcta = gpd.read_file(zcta_shapefile)
    zip_col = next(
        (c for c in ["ZCTA5CE20", "ZCTA5CE10", "ZCTA5CE"] if c in zcta.columns),
        next(c for c in zcta.columns if "ZCTA5" in c.upper()),
    )
    zcta = zcta[[zip_col, "geometry"]].rename(columns={zip_col: "zip"})
    zcta["zip"] = norm_zip(zcta["zip"])
    zcta = zcta.dropna(subset=["geometry"]).drop_duplicates("zip")
    if zcta.crs is None:
        zcta = zcta.set_crs("EPSG:4326", allow_override=True)

    # Restrict to the storm's footprint so the spatial join stays cheap.
    bbox = box(latlon[:, 1].min() - 0.75, latlon[:, 0].min() - 0.75,
               latlon[:, 1].max() + 0.75, latlon[:, 0].max() + 0.75)
    local = zcta[zcta.intersects(bbox)]
    if local.empty:
        local = zcta

    points = gpd.GeoDataFrame(
        {"stay_h3": cells},
        geometry=gpd.points_from_xy(latlon[:, 1], latlon[:, 0]),
        crs=local.crs,
    )
    joined = gpd.sjoin(points, local, how="left", predicate="within")
    joined = (
        joined[["stay_h3", "zip"]]
        .sort_values(["stay_h3", "zip"], na_position="last")
        .drop_duplicates("stay_h3", keep="first")
    )
    return dict(zip(joined["stay_h3"], joined["zip"]))


def run_aggregate_stage(args: argparse.Namespace) -> None:
    """Rebuild the panel, the ZIP intensity map and the evacuation rates."""
    import duckdb

    lbs_base = Path(args.lbs_base_dir).expanduser().resolve()
    if not lbs_base.is_dir():
        raise SystemExit(f"--lbs-base-dir does not exist: {lbs_base}")
    if not args.zcta_shapefile:
        raise SystemExit("--zcta-shapefile is required when --stage aggregate")

    zcta_shapefile = Path(args.zcta_shapefile).expanduser().resolve()
    agg_dir = Path(args.agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)

    # The intensity assignment is a fixed input derived from HURDAT2 upstream
    # of this script; the shipped copy is used unless one is given explicitly.
    intensity_path = (
        Path(args.intensity_lookup) if args.intensity_lookup
        else agg_dir / AGG_FILES["zip_intensity"]
    )
    if not intensity_path.exists():
        raise SystemExit(
            f"Intensity lookup not found: {intensity_path}\n"
            "Pass --intensity-lookup, or restore the copy shipped with the "
            "repository. See data/hurdat2/README.md for how it is derived."
        )

    log("Loading ZIP centroids ...")
    coords = build_zip_centroids(zcta_shapefile)
    log(f"  {len(coords):,} ZIP centroids")

    intensity = pd.read_csv(intensity_path, dtype={"home_zip": str})
    intensity["home_zip"] = norm_zip(intensity["home_zip"])
    intensity = intensity.dropna(subset=["disaster", "home_zip", "int_bin"])
    intensity = intensity.drop_duplicates(["disaster", "home_zip"])
    log(f"  intensity lookup: {len(intensity):,} disaster-ZIP pairs "
        f"from {intensity_path.name}")

    # Home ZIP and income group per user, from the evacuation cohort.
    cohort = pd.read_csv(args.evacuation_status, dtype={"pre_crisis_home_zip": str})
    cohort = cohort.loc[
        (cohort["threshold_minutes"] == MAIN_THRESHOLD)
        & cohort["pre_crisis_home_zip"].notna()
    ]
    person = (
        cohort[["pid", "disaster", "pre_crisis_home_zip", "evacuated"]]
        .drop_duplicates(["pid", "disaster"])
        .rename(columns={"pre_crisis_home_zip": "home_zip"})
    )
    person["home_zip"] = norm_zip(person["home_zip"])
    person = person.merge(intensity, on=["disaster", "home_zip"], how="left")

    income = pd.read_csv(args.income_lookup, dtype={"home_zip": str})
    income["home_zip"] = norm_zip(income["home_zip"])
    person = person.merge(income, on="home_zip", how="left")
    median_income = float(person["median_income"].median())
    person["income_group"] = np.where(
        person["median_income"] >= median_income, "High", "Low"
    )
    person.loc[person["median_income"].isna(), "income_group"] = np.nan
    log(f"  cohort: {len(person):,} users; median ZIP income {median_income:,.0f}")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.duckdb_threads}")

    panel_parts, evac_parts = [], []
    try:
        for cfg in DISASTERS:
            log(f"Computing daily Rg for {cfg['label']} ...")
            dataset_dir = lbs_base / cfg["dataset"]

            # Which H3 cells appear in this window, so only those are mapped.
            glob_path = str(dataset_dir / HOME_WORK_STAYS / "*.parquet")
            day_lo = (pd.Timestamp(cfg["event_date"]) + pd.Timedelta(days=T_LO)).date()
            day_hi = (pd.Timestamp(cfg["event_date"]) + pd.Timedelta(days=T_HI)).date()
            cells = con.execute(f"""
                SELECT DISTINCT CAST(h3_index AS VARCHAR) AS stay_h3
                FROM read_parquet('{glob_path}')
                WHERE CAST(local_time AS DATE) >= DATE '{day_lo}'
                  AND CAST(local_time AS DATE) <= DATE '{day_hi}'
                  AND h3_index IS NOT NULL
            """).df()["stay_h3"].dropna().astype(str).tolist()
            log(f"  {len(cells):,} distinct H3 cells")
            h3_to_zip = map_h3_to_zip(sorted(cells), zcta_shapefile)

            rg = compute_daily_rg(con, dataset_dir, cfg["event_date"], coords, h3_to_zip)
            if rg.empty:
                log("  no radius of gyration rows; skipping")
                continue
            log(f"  {len(rg):,} user-day values")

            rg = rg.merge(
                person.loc[person["disaster"] == cfg["label"],
                           ["pid", "int_bin", "income_group"]],
                on="pid", how="inner",
            ).dropna(subset=["int_bin", "income_group", "rg_km"])

            cell = (
                rg.groupby(["t", "int_bin", "income_group"], as_index=False)
                .agg(mean_rog=("rg_km", "mean"), n=("rg_km", "count"))
            )
            cell.insert(0, "disaster", cfg["label"])
            panel_parts.append(cell)

            # Evacuation rate by intensity bin, for the final panel.
            storm = person.loc[person["disaster"] == cfg["label"]].dropna(subset=["int_bin"])
            if not storm.empty:
                rate = (
                    storm.groupby("int_bin", as_index=False)
                    .agg(eligible=("pid", "nunique"), evacuated=("evacuated", "sum"))
                )
                rate["evacuation_rate"] = 100.0 * rate["evacuated"] / rate["eligible"]
                rate.insert(0, "disaster", cfg["label"])
                evac_parts.append(rate)
    finally:
        con.close()

    if panel_parts:
        pd.concat(panel_parts, ignore_index=True).to_csv(
            agg_dir / AGG_FILES["panel"], index=False
        )
        log(f"  wrote {agg_dir / AGG_FILES['panel']}")
    if evac_parts:
        pd.concat(evac_parts, ignore_index=True).to_csv(
            agg_dir / AGG_FILES["evac_rate"], index=False
        )
        log(f"  wrote {agg_dir / AGG_FILES['evac_rate']}")


# --------------------------------------------------------------------------
# Stage 2: regressions
# --------------------------------------------------------------------------

def fit_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate equations (2) and (3) of Supplementary Note 4.

    Returns ``(table_s2, table_s3)``. Both use disaster fixed effects and HC3
    standard errors, and are fitted on the storms in
    :data:`REGRESSION_DISASTERS`.
    """
    import statsmodels.formula.api as smf

    # The panel is smoothed before fitting, matching the series drawn in the
    # figure, so the estimates describe the curves the figure shows.
    reg = smooth_panel(panel)
    reg = reg.loc[reg["disaster"].isin(REGRESSION_DISASTERS)].copy()
    if reg.empty:
        raise SystemExit(
            "No rows available for the regressions. Expected disasters: "
            + ", ".join(REGRESSION_DISASTERS)
        )

    reg["post"] = (reg["t"] >= 0).astype(int)
    reg["low_income"] = (reg["income_group"] == "Low").astype(int)

    # Equation (2): the post-landfall change within each intensity bin.
    rows = []
    for label in INTENSITY_ORDER:
        sub = reg.loc[reg["int_bin"] == label]
        if len(sub) < 10 or sub["post"].nunique() < 2:
            log(f"  skipping bin with insufficient variation: {label}")
            continue
        model = smf.ols(
            "mean_rog ~ post * low_income + C(disaster)", data=sub
        ).fit(cov_type="HC3")
        rows.append({
            "intensity_bin": label,
            "alpha1_km": model.params["post"],
            "p": model.pvalues["post"],
            "n": len(sub),
            "interaction_low_income_km": model.params.get("post:low_income", np.nan),
            "interaction_p": model.pvalues.get("post:low_income", np.nan),
        })
    table_s2 = pd.DataFrame(rows)

    # Equation (3): intensity rank as an ordered covariate.
    reg["int_rank"] = reg["int_bin"].map(intensity_rank)
    reg = reg.loc[reg["int_rank"] < 99]
    pooled = smf.ols(
        "mean_rog ~ post * low_income * int_rank + C(disaster)", data=reg
    ).fit(cov_type="HC3")

    terms = {
        "post": "Post",
        "post:int_rank": r"Post $\times$ k (intensity rank)",
        "low_income": "Low income",
        "post:low_income": r"Post $\times$ low income",
        "low_income:int_rank": r"Low income $\times$ k",
        "post:low_income:int_rank": r"Post $\times$ low income $\times$ k",
    }
    rows = []
    for term, label in terms.items():
        if term not in pooled.params.index:
            continue
        lo, hi = pooled.conf_int().loc[term]
        rows.append({
            "term": label,
            "coef": pooled.params[term],
            "std_err": pooled.bse[term],
            "z": pooled.tvalues[term],
            "p": pooled.pvalues[term],
            "ci_low": lo,
            "ci_high": hi,
        })
    table_s3 = pd.DataFrame(rows)
    return table_s2, table_s3


def write_table_s2(table: pd.DataFrame, out_dir: Path) -> None:
    """Write table S2 as CSV, and echo it to stdout."""
    table.to_csv(out_dir / "table_s2.csv", index=False)

    log("\nTable S2: post-landfall change in mean Rg, by intensity bin")
    for row in table.itertuples(index=False):
        p = "<0.001" if row.p < 0.001 else f"{row.p:.3f}"
        log(f"  {row.intensity_bin:28s} a1 = {row.alpha1_km:+.3f} km   "
            f"p = {p:>6s}   n = {int(row.n)}")


def write_table_s3(table: pd.DataFrame, out_dir: Path) -> None:
    """Write table S3 as CSV, and echo it to stdout."""
    table.to_csv(out_dir / "table_s3.csv", index=False)

    log("\nTable S3: pooled estimates with intensity rank as an ordered covariate")
    for row in table.itertuples(index=False):
        p = "<0.001" if row.p < 0.001 else f"{row.p:.3f}"
        log(f"  {row.term:42s} {row.coef:+.3f}  (SE {row.std_err:.3f}, "
            f"z {row.z:+.3f}, p {p})")


# --------------------------------------------------------------------------
# Stage 2: figure
# --------------------------------------------------------------------------

def plot_intensity_map(ax, geometries, disaster: str) -> None:
    """Shade each residential ZIP code by the wind intensity it experienced."""
    for label in INTENSITY_ORDER:
        subset = geometries.loc[geometries["int_bin"] == label]
        if not subset.empty:
            subset.plot(ax=ax, color=INTENSITY_COLORS[label],
                        linewidth=0.15, edgecolor="white", zorder=2)

    try:
        import contextily as ctx

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                        attribution=False, zorder=1)
    except Exception:
        # A basemap is decorative; the choropleth stands on its own without it.
        ax.set_facecolor("#eaeaea")

    ax.set_aspect("auto")
    ax.axis("off")


def plot_figure_3(
    panel: pd.DataFrame, zip_intensity: pd.DataFrame, evac_rate: pd.DataFrame,
    plots_dir: Path, dpi: int, zcta_shapefile: Path | None,
) -> None:
    """Assemble figure 3 from the shared aggregates."""
    panel = smooth_panel(panel)
    disasters = [d for d in FIGURE_ORDER if d in set(panel["disaster"])]

    geometries = {}
    if zcta_shapefile is not None and zcta_shapefile.exists() and not zip_intensity.empty:
        import geopandas as gpd

        zcta = gpd.read_file(zcta_shapefile)
        zip_col = next(
            (c for c in ["home_zip", "ZCTA5CE20", "ZCTA5CE10", "ZCTA5CE"]
             if c in zcta.columns),
            None,
        )
        if zip_col is None:
            zip_col = next((c for c in zcta.columns if "ZCTA5" in c.upper()), None)
        if zip_col is None:
            raise SystemExit(
                f"No ZIP code column found in {zcta_shapefile}. Expected one of "
                "home_zip, ZCTA5CE20, ZCTA5CE10 or ZCTA5CE; found: "
                + ", ".join(zcta.columns)
            )
        zcta = zcta[[zip_col, "geometry"]].rename(columns={zip_col: "home_zip"})
        zcta["home_zip"] = norm_zip(zcta["home_zip"])
        if zcta.crs is None:
            zcta = zcta.set_crs(epsg=4269, allow_override=True)

        for disaster in disasters:
            wanted = zip_intensity.loc[zip_intensity["disaster"] == disaster]
            merged = zcta.merge(wanted[["home_zip", "int_bin"]], on="home_zip", how="inner")
            merged = merged.loc[~merged.geometry.is_empty]
            if not merged.empty:
                geometries[disaster] = merged.to_crs(epsg=3857)
    else:
        log("  ZIP boundaries unavailable; intensity maps will be omitted")

    plt.rcParams.update({
        "font.size": FS, "axes.labelsize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS,
    })

    # Six cells in a 2x3 grid: five disasters, then the evacuation panels.
    # Each cell holds a line/map pair, sized so both are roughly square, as in
    # the published figure.
    fig = plt.figure(figsize=(30, 10), dpi=dpi)
    outer = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.20)

    for idx, disaster in enumerate(disasters[:5]):
        row, col = divmod(idx, 3)
        inner = outer[row, col].subgridspec(1, 2, wspace=0.05)
        ax_line = fig.add_subplot(inner[0, 0])
        ax_map = fig.add_subplot(inner[0, 1])

        sub = panel.loc[panel["disaster"] == disaster]
        present = [b for b in INTENSITY_ORDER if b in set(sub["int_bin"])]

        # One line per intensity bin, averaging over income groups.
        for label in present:
            series = (
                sub.loc[sub["int_bin"] == label]
                .groupby("t", as_index=False)
                .apply(lambda g: pd.Series({
                    "mean_rog": np.average(g["mean_rog"], weights=g["n"])
                    if g["n"].sum() else np.nan
                }), include_groups=False)
                .sort_values("t")
            )
            ax_line.plot(series["t"], series["mean_rog"], lw=2.6,
                         color=INTENSITY_COLORS[label],
                         label=INTENSITY_SHORT[label])

        ax_line.axvline(0, color="black", lw=1.3, ls="--", alpha=0.6)
        ax_line.set_xlim(T_LO, T_HI)
        ax_line.set_xticks([-6, -3, 0, 3, 6])
        ax_line.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_line.tick_params(labelsize=FS - 4)
        ax_line.grid(True, alpha=0.2, ls="--")
        ax_line.set_xlabel("Day relative to landfall", fontsize=FS - 2)
        ax_line.set_ylabel("Mean $R_g$ (km)", fontsize=FS - 2)
        ax_line.text(0.03, 0.03, disaster.split(" (")[0], transform=ax_line.transAxes,
                     ha="left", va="bottom", fontsize=FS - 4,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))
        if len(present) > 1:
            ax_line.legend(frameon=False, fontsize=FS - 9, loc="upper right",
                           handlelength=1.6, labelspacing=0.3)

        if disaster in geometries:
            plot_intensity_map(ax_map, geometries[disaster], disaster)
        else:
            ax_map.axis("off")

        ax_line.text(-0.18, 1.04, "ABCDE"[idx], transform=ax_line.transAxes,
                     fontsize=FS, fontweight="bold", va="top")

    # Final cell: evacuation rate against intensity, per storm and pooled.
    inner = outer[1, 2].subgridspec(1, 2, wspace=0.35)
    ax_storm = fig.add_subplot(inner[0, 0])
    ax_pooled = fig.add_subplot(inner[0, 1])

    if not evac_rate.empty:
        evac_rate = evac_rate.copy()
        evac_rate["rank"] = evac_rate["int_bin"].map(intensity_rank)
        evac_rate = evac_rate.loc[evac_rate["rank"] < 99]

        storm_colors = {
            "Harvey (2017)": "#4C72B0", "Irma (2017)": "#8172B2",
            "Florence (2018)": "#3BC9DB", "Michael (2018)": "#7D3C4A",
            "Imelda (2019)": "#8C8C8C",
        }
        for disaster in FIGURE_ORDER:
            series = evac_rate.loc[evac_rate["disaster"] == disaster].sort_values("rank")
            if series.empty:
                continue
            ax_storm.plot(series["rank"], series["evacuation_rate"], marker="o",
                          markersize=5, lw=1.8,
                          color=storm_colors.get(disaster, "#333333"),
                          label=disaster.split(" (")[0])
        ax_storm.set_xlabel("Disaster Intensity", fontsize=FS - 6)
        ax_storm.set_ylabel("Evacuation rate (%)", fontsize=FS - 6)
        ax_storm.set_xticks(range(len(INTENSITY_ORDER)))
        ax_storm.tick_params(labelsize=FS - 8)
        ax_storm.grid(True, alpha=0.2, ls="--")
        ax_storm.legend(frameon=False, fontsize=FS - 11)

        # Pooled across storms, weighting each bin by its eligible population.
        pooled = (
            evac_rate.groupby("rank", as_index=False)
            .agg(eligible=("eligible", "sum"), evacuated=("evacuated", "sum"))
        )
        pooled["rate"] = 100.0 * pooled["evacuated"] / pooled["eligible"]
        # Binomial standard error on the pooled rate.
        pooled["err"] = 100.0 * np.sqrt(
            (pooled["evacuated"] / pooled["eligible"])
            * (1 - pooled["evacuated"] / pooled["eligible"])
            / pooled["eligible"]
        )
        ax_pooled.bar(
            pooled["rank"], pooled["rate"], yerr=pooled["err"], capsize=4,
            color=[INTENSITY_COLORS[INTENSITY_ORDER[int(r)]] for r in pooled["rank"]],
            edgecolor="black", linewidth=0.6,
        )
        ax_pooled.set_xlabel("Disaster Intensity", fontsize=FS - 6)
        ax_pooled.set_xticks(range(len(INTENSITY_ORDER)))
        ax_pooled.tick_params(labelsize=FS - 8)
        ax_pooled.grid(True, axis="y", alpha=0.2, ls="--")

    ax_storm.text(-0.22, 1.04, "F", transform=ax_storm.transAxes,
                  fontsize=FS, fontweight="bold", va="top")

    out = plots_dir / "figure_3.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def run_figures_stage(args: argparse.Namespace) -> None:
    """Produce the figure and both regression tables from the aggregates."""
    agg_dir = Path(args.agg_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    panel = read_aggregate(agg_dir, "panel")
    for column in ["disaster", "t", "int_bin", "income_group", "mean_rog", "n"]:
        if column not in panel.columns:
            raise SystemExit(
                f"{AGG_FILES['panel']} is missing the '{column}' column; found: "
                + ", ".join(panel.columns)
            )

    if not args.skip_tables:
        log("Estimating regressions ...")
        table_s2, table_s3 = fit_tables(panel)
        write_table_s2(table_s2, plots_dir)
        write_table_s3(table_s3, plots_dir)

    if not args.skip_figure:
        log("\nRendering figure 3 ...")
        zip_intensity = read_aggregate(agg_dir, "zip_intensity", required=False)
        evac_rate = read_aggregate(agg_dir, "evac_rate", required=False)

        # Prefer an explicit shapefile, otherwise fall back to the ZIP
        # boundaries bundled with the repository.
        if args.zcta_shapefile:
            geometry_path = Path(args.zcta_shapefile).expanduser()
        else:
            geometry_path = agg_dir / FOOTPRINT_GEOMETRY

        plot_figure_3(panel, zip_intensity, evac_rate, plots_dir, args.dpi, geometry_path)


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
        help="'figures' (default) produces the figure and tables from the "
             "shared aggregates; 'aggregate' rebuilds them from the raw panel.",
    )
    parser.add_argument(
        "--data-dir", default=str(here / "data"),
        help="Repository data directory (default: %(default)s).",
    )
    parser.add_argument(
        "--agg-dir", default=None,
        help="Directory holding the aggregate tables "
             "(default: <data-dir>/rg_variation).",
    )
    parser.add_argument(
        "--plots-dir", default=str(here / "plots"),
        help="Directory for the figure and tables (default: %(default)s).",
    )
    parser.add_argument("--dpi", type=int, default=300,
                        help="Figure resolution (default: %(default)s).")
    parser.add_argument("--skip-figure", action="store_true",
                        help="Produce only the regression tables.")
    parser.add_argument("--skip-tables", action="store_true",
                        help="Produce only the figure.")
    parser.add_argument(
        "--zcta-shapefile", default=None,
        help="Full TIGER/Line ZCTA shapefile. Required for --stage aggregate. "
             "The figure stage does not need it: it uses the ZIP boundaries "
             "bundled in the data directory, and this option only overrides "
             "them.",
    )

    raw = parser.add_argument_group("raw data options (--stage aggregate only)")
    raw.add_argument(
        "--lbs-base-dir",
        help="Directory containing the processed stay-point datasets. The raw "
             "LBS data is proprietary and is not distributed with this "
             "repository.",
    )
    raw.add_argument(
        "--intensity-lookup", default=None,
        help="CSV of disaster, home_zip and int_bin derived from HURDAT2. "
             "Defaults to the copy shipped in the data directory; see "
             "data/hurdat2/README.md for how it was derived.",
    )
    raw.add_argument(
        "--evacuation-status",
        help="CSV of per-user evacuation status from the evacuation pipeline, "
             "with columns pid, disaster, pre_crisis_home_zip, evacuated and "
             "threshold_minutes.",
    )
    raw.add_argument(
        "--income-lookup",
        help="CSV of home_zip and median_income (ACS table B19013_001E).",
    )
    raw.add_argument(
        "--duckdb-threads", type=int, default=8,
        help="Threads used by DuckDB (default: %(default)s).",
    )

    args = parser.parse_args(argv)
    if args.agg_dir is None:
        args.agg_dir = str(Path(args.data_dir) / "rg_variation")

    if args.stage == "aggregate":
        missing = [
            name for name, value in [
                ("--lbs-base-dir", args.lbs_base_dir),
                ("--evacuation-status", args.evacuation_status),
                ("--income-lookup", args.income_lookup),
            ] if not value
        ]
        if missing:
            parser.error(f"{', '.join(missing)} required when --stage aggregate")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "aggregate":
        run_aggregate_stage(args)
    else:
        run_figures_stage(args)
    log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
