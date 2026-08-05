#!/usr/bin/env python3
"""
01_processing_cdr.py -- processing of call detail records (CDR).

    figure_s1.png  Network construction and CDR-vs-census sanity checks: edges
                   against the interaction threshold, the degree distribution,
                   the ZIP-level expansion factor, inferred homes against the
                   night-events threshold, unique locations visited, and a map
                   of the expansion factor across the Bay Area.
    figure_s2.png  Sensitivity of the degree distribution and of three social
                   capital metrics (clustering, support ratio, economic
                   connectedness) to the interaction threshold m.

A user's home is the cell where they log the most nighttime (20:00-06:00
local) voice activity, kept if that cell accrues at least 28 such events; the
home cell is then mapped to its ZCTA5 ZIP code. The interaction network is
undirected: two users are linked if their combined call-plus-SMS count over
the study window is at least m (m=3 throughout the main analysis, swept over
{1,2,3,5,7,10} for the sensitivity panels). The expansion factor is a ZIP
code's ACS population divided by the number of distinct users CDR detects
there, a proxy for sample coverage.

Clustering is each user's local clustering coefficient over the full
(cross-ZIP) network, averaged over the residents of a ZIP; support ratio is
the share of within-ZIP edges whose endpoints share a common neighbour;
economic connectedness follows Chetty et al.: for each below-median-income
user, the mean share of their contacts who are above-median income, doubled
and averaged over the ZIP's below-median residents.

The raw CDR traces cannot be redistributed, so the script runs in two stages:
``--stage figures`` (default) draws from the shared aggregates in ``data/`` and
needs no raw data; ``--stage aggregate`` rebuilds those aggregates from the
proprietary CDR panel. Run with --help for options.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Non-interactive backend: the script only ever writes files.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402


# --------------------------------------------------------------------------
# Study configuration
# --------------------------------------------------------------------------

#: Study window for the CDR panel.
START_DATE = "2018-12-01"
END_DATE = "2019-06-30"

#: Local timezone and the nighttime hours used to infer a home cell.
TZ_LOCAL = "America/Los_Angeles"
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 6

#: A cell must accrue at least this many nighttime events to count as home.
MIN_HOME_NIGHT_EVENTS = 28
#: Night-events thresholds swept for figure S1, panel D.
HOME_THRESHOLDS = [7, 14, 21, 28, 35, 42, 56, 70]

#: An edge must accrue at least this many combined call+SMS interactions.
MIN_EDGE_INTERACTIONS = 3
#: Interaction thresholds swept for figures S1 (panel A) and S2.
EDGE_THRESHOLDS = [1, 2, 3, 5, 7, 10]

#: Bay Area bounding box used for the figure S1 choropleth.
BAY_AREA_BBOX = {"lat_min": 37.0, "lat_max": 38.3, "lon_min": -123.0, "lon_max": -121.5}

#: Panels of figure S2 and the x-axis range each is drawn over.
SC_PANELS = [
    ("clust", "Clustering", (0.0, 0.7)),
    ("sr", "Support Ratio", (0.0, 1.0)),
    ("ec", "Economic Connectedness", (0.0, 2.0)),
]

#: KDE bandwidth used for every social-capital density in figure S2.
KDE_BW_METHOD = 0.15

FS = 20

#: Aggregate tables written by ``--stage aggregate``.
AGG_FILES = {
    "edges_by_threshold": "s1_edges_by_threshold.csv",
    "degree_dist": "s1_degree_distribution.csv",
    "expansion_factor_kde": "s1_expansion_factor_kde.csv",
    "expansion_factor_zip": "s1_expansion_factor_by_zip.csv",
    "homes_by_threshold": "s1_homes_by_threshold.csv",
    "unique_locations_dist": "s1_unique_locations_distribution.csv",
    "strength_by_threshold": "s2_strength_by_threshold.csv",
    "soccap_kde": "s2_soccap_kde_by_threshold.csv",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def log(message: str) -> None:
    """Print a progress message immediately."""
    print(message, flush=True)


def zfill_zip(values) -> pd.Series:
    """Normalise ZIP/ZCTA5 codes to zero-padded five-character strings."""
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(5)
    )


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """Fail early with a readable message if an aggregate table is malformed."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(df.columns)}"
        )


def read_aggregate(agg_dir: Path, key: str, required: bool = True) -> pd.DataFrame:
    """Load one aggregate table, with a message pointing at the other stage."""
    path = agg_dir / AGG_FILES[key]
    if not path.exists():
        if not required:
            return pd.DataFrame()
        raise SystemExit(
            f"Missing aggregate table: {path}\n"
            f"Run `python {Path(__file__).name} --stage aggregate` with access to "
            f"the raw CDR panel to regenerate it, or restore the file shipped "
            f"with the repository."
        )
    dtype = {"zip": str} if key in ["expansion_factor_zip"] else {}
    return pd.read_csv(path, dtype=dtype)


def kde_on_grid(values: np.ndarray, xlim: tuple[float, float], n: int = 500,
                bw_method: float | str = KDE_BW_METHOD) -> tuple[np.ndarray, np.ndarray] | None:
    """Evaluate a Gaussian KDE on a fixed grid, or None if too few points."""
    from scipy.stats import gaussian_kde

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5 or np.unique(values).size < 2:
        return None
    xgrid = np.linspace(xlim[0], xlim[1], n)
    return xgrid, gaussian_kde(values, bw_method=bw_method)(xgrid)


# --------------------------------------------------------------------------
# Stage 1: aggregation from the raw CDR panel
# --------------------------------------------------------------------------

def iter_dates(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    d = d0
    while d <= d1:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def load_old2new_cells(path: Path) -> dict[int, int]:
    """Map raw carrier cell ids to a dense, normalised cell id."""
    import lzma
    import pickle

    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def to_local_naive_from_utc(series: pd.Series, tz: str) -> pd.Series:
    """Convert a UTC datetime Series to tz-naive local time."""
    s = pd.to_datetime(series)
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert(tz).dt.tz_localize(None)


def is_night_local(local_dt: pd.Series) -> pd.Series:
    h = local_dt.dt.hour
    return (h >= NIGHT_START_HOUR) | (h < NIGHT_END_HOUR)


def build_home_cells(
    voice_dir: Path, cell_map_path: Path, work_dir: Path,
) -> pd.DataFrame:
    """Infer each user's home cell from nighttime voice activity.

    A user's home is the cell accruing the most nighttime events, subject to a
    minimum of :data:`MIN_HOME_NIGHT_EVENTS`. Also writes the per-user event
    count at their top cell, before thresholding, so the night-events
    sensitivity sweep in figure S1 does not need to reprocess the raw panel.
    """
    import duckdb

    old2new = load_old2new_cells(cell_map_path)
    visits_dir = work_dir / "home_visits_daily"
    visits_dir.mkdir(parents=True, exist_ok=True)

    for d in iter_dates(START_DATE, END_DATE):
        out_path = visits_dir / f"home_visits_{d}.parquet"
        if out_path.exists():
            continue
        vp = voice_dir / f"voice_{d}_parquet.gzip"
        if not vp.exists():
            continue

        df = pd.read_parquet(vp, columns=["A_MSISDN", "START_TIME_GMT", "CELL_0"])
        local_dt = to_local_naive_from_utc(df["START_TIME_GMT"], TZ_LOCAL)
        df = df.loc[is_night_local(local_dt), ["A_MSISDN", "CELL_0"]].copy()

        cell0 = pd.to_numeric(df["CELL_0"], errors="coerce")
        df = df.loc[cell0.notna()].copy()
        df["cell"] = cell0.loc[cell0.notna()].astype(np.int64).map(old2new)
        df = df.drop(columns=["CELL_0"]).dropna(subset=["cell"])
        df["cell"] = df["cell"].astype(np.int32)

        part = df.groupby(["A_MSISDN", "cell"], as_index=False).size()
        part.rename(columns={"size": "n_visits"}).to_parquet(out_path, index=False)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute(f"""
        CREATE OR REPLACE TABLE home_visits AS
        SELECT A_MSISDN AS msisdn, cell, SUM(n_visits)::BIGINT AS n_visits
        FROM read_parquet('{visits_dir / "home_visits_*.parquet"}')
        GROUP BY 1, 2
    """)
    top_cell = con.execute("""
        SELECT msisdn, cell AS home_cell, n_visits AS n_night_events
        FROM (
            SELECT msisdn, cell, n_visits,
                   ROW_NUMBER() OVER (
                       PARTITION BY msisdn ORDER BY n_visits DESC, cell ASC
                   ) AS rn
            FROM home_visits
        )
        WHERE rn = 1
    """).df()
    con.close()

    log(f"  users with >=1 nighttime event: {len(top_cell):,}")
    return top_cell


def map_cells_to_zip(home_cells: pd.DataFrame, cells_csv: Path, zcta_shapefile: Path) -> pd.DataFrame:
    """Assign each inferred home cell to the ZCTA5 ZIP code containing it."""
    import geopandas as gpd
    from shapely.geometry import Point

    cells_df = pd.read_csv(cells_csv, usecols=["cell_id", "latitude", "longitude"])
    home_ids = home_cells.loc[
        home_cells["n_night_events"] >= MIN_HOME_NIGHT_EVENTS, "home_cell"
    ].unique()
    cell_pts = cells_df[cells_df["cell_id"].isin(home_ids)].copy()
    cell_pts["geometry"] = [
        Point(xy) for xy in zip(cell_pts["longitude"], cell_pts["latitude"])
    ]

    zcta = gpd.read_file(zcta_shapefile)
    zip_col = next(
        (c for c in ["ZCTA5CE20", "ZCTA5CE10", "ZCTA5CE"] if c in zcta.columns),
        next(c for c in zcta.columns if "ZCTA5" in c.upper()),
    )
    zcta = zcta[[zip_col, "geometry"]].rename(columns={zip_col: "zip"})
    zcta["zip"] = zfill_zip(zcta["zip"])
    if zcta.crs is not None and zcta.crs.to_epsg() != 4326:
        zcta = zcta.to_crs(epsg=4326)

    points = gpd.GeoDataFrame(
        cell_pts[["cell_id"]], geometry=cell_pts["geometry"], crs="EPSG:4326"
    )
    joined = gpd.sjoin(points, zcta, how="left", predicate="within")
    joined = joined[["cell_id", "zip"]].drop_duplicates("cell_id")

    home = home_cells.loc[home_cells["n_night_events"] >= MIN_HOME_NIGHT_EVENTS].merge(
        joined.rename(columns={"cell_id": "home_cell"}), on="home_cell", how="left"
    )
    log(f"  home ZIP null rate: {home['zip'].isna().mean():.3f}")
    return home.dropna(subset=["zip"])[["msisdn", "home_cell", "zip"]]


def build_edges(voice_dir: Path, sms_dir: Path, users_zip: pd.DataFrame, work_dir: Path) -> Path:
    """Aggregate the undirected call+SMS interaction count for every user pair."""
    import duckdb

    users_path = work_dir / "users_with_zip.parquet"
    users_zip.to_parquet(users_path, index=False)

    edges_dir = work_dir / "edges_daily"
    edges_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    users_tbl = f"read_parquet('{users_path.as_posix()}')"

    for d in iter_dates(START_DATE, END_DATE):
        out_path = edges_dir / f"edges_{d}.parquet"
        if out_path.exists():
            continue
        vp, sp = voice_dir / f"voice_{d}_parquet.gzip", sms_dir / f"sms_{d}_parquet.gzip"
        if not vp.exists() and not sp.exists():
            continue

        voice_sql = "SELECT NULL::BIGINT u, NULL::BIGINT v, 0::BIGINT n_voice WHERE FALSE"
        if vp.exists():
            voice_sql = f"""
                SELECT LEAST(v.A_MSISDN, v.B_MSISDN)::BIGINT AS u,
                       GREATEST(v.A_MSISDN, v.B_MSISDN)::BIGINT AS v,
                       COUNT(*)::BIGINT AS n_voice
                FROM read_parquet('{vp.as_posix()}') v
                JOIN {users_tbl} ua ON v.A_MSISDN = ua.msisdn
                JOIN {users_tbl} ub ON v.B_MSISDN = ub.msisdn
                WHERE v.A_MSISDN != v.B_MSISDN
                GROUP BY 1, 2
            """
        sms_sql = "SELECT NULL::BIGINT u, NULL::BIGINT v, 0::BIGINT n_sms WHERE FALSE"
        if sp.exists():
            sms_sql = f"""
                SELECT LEAST(s.A_MSISDN, s.B_MSISDN)::BIGINT AS u,
                       GREATEST(s.A_MSISDN, s.B_MSISDN)::BIGINT AS v,
                       COUNT(*)::BIGINT AS n_sms
                FROM read_parquet('{sp.as_posix()}') s
                JOIN {users_tbl} ua ON s.A_MSISDN = ua.msisdn
                JOIN {users_tbl} ub ON s.B_MSISDN = ub.msisdn
                WHERE s.A_MSISDN != s.B_MSISDN
                GROUP BY 1, 2
            """
        con.execute(f"""
            COPY (
                WITH voice AS ({voice_sql}), sms AS ({sms_sql})
                SELECT COALESCE(voice.u, sms.u) AS u, COALESCE(voice.v, sms.v) AS v,
                       COALESCE(voice.n_voice, 0)::BIGINT AS n_voice,
                       COALESCE(sms.n_sms, 0)::BIGINT AS n_sms
                FROM voice FULL OUTER JOIN sms USING (u, v)
            ) TO '{out_path.as_posix()}' (FORMAT PARQUET)
        """)

    edges_all_path = work_dir / "edges_all.parquet"
    con.execute(f"""
        COPY (
            SELECT u, v, SUM(n_voice)::BIGINT AS n_voice, SUM(n_sms)::BIGINT AS n_sms
            FROM read_parquet('{edges_dir / "edges_*.parquet"}')
            GROUP BY 1, 2
        ) TO '{edges_all_path.as_posix()}' (FORMAT PARQUET)
    """)
    con.close()
    return edges_all_path


def build_unique_locations(voice_dir: Path, cell_map_path: Path, node_ids: np.ndarray) -> np.ndarray:
    """Distinct cells visited per user, restricted to users in the network."""
    import duckdb

    old2new = load_old2new_cells(cell_map_path)
    map_df = pd.DataFrame(
        {"old_cell": list(old2new.keys()), "cell": list(old2new.values())}
    ).dropna().astype("int64")

    con = duckdb.connect()
    con.register("cell_map_df", map_df)
    con.register("users_df", pd.DataFrame({"msisdn": node_ids.astype("int64")}))
    out = con.execute(f"""
        WITH v AS (
            SELECT CAST(A_MSISDN AS BIGINT) AS msisdn, TRY_CAST(CELL_0 AS BIGINT) AS old_cell
            FROM read_parquet('{voice_dir / "voice_*_parquet.gzip"}')
            WHERE CELL_0 IS NOT NULL
        ),
        vm AS (SELECT v.msisdn, m.cell FROM v JOIN cell_map_df m ON v.old_cell = m.old_cell)
        SELECT vm.msisdn, COUNT(DISTINCT vm.cell) AS s
        FROM vm JOIN users_df u ON vm.msisdn = u.msisdn
        GROUP BY 1
    """).df()
    con.close()
    return out["s"].to_numpy()


def local_clustering(u: np.ndarray, v: np.ndarray, n_nodes: int) -> np.ndarray:
    """Local clustering coefficient of every node in an undirected edge list."""
    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    order = np.lexsort((dst, src))
    nbrs = dst[order].astype(np.int64)
    deg = np.bincount(src[order], minlength=n_nodes).astype(np.int64)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])

    contrib = np.zeros(n_nodes, dtype=np.int64)
    for a, b in zip(u.tolist(), v.tolist()):
        na = nbrs[indptr[a]:indptr[a + 1]]
        nb = nbrs[indptr[b]:indptr[b + 1]]
        c = int(np.intersect1d(na, nb, assume_unique=True).size)
        if c:
            contrib[a] += c
            contrib[b] += c

    triangles = contrib / 2.0
    deg_f = deg.astype(np.float64)
    pairs = deg_f * (deg_f - 1.0) / 2.0
    return np.where(pairs > 0, triangles / pairs, 0.0)


def clustering_by_zip(u: np.ndarray, v: np.ndarray, n_nodes: int, zips: np.ndarray) -> np.ndarray:
    """Mean local clustering coefficient of the residents of each ZIP."""
    clust = local_clustering(u, v, n_nodes)
    return pd.DataFrame({"zip": zips, "clust": clust}).groupby("zip")["clust"].mean().to_numpy()


def support_ratio_by_zip(u: np.ndarray, v: np.ndarray, zips: np.ndarray) -> np.ndarray:
    """Share of within-ZIP edges whose endpoints share a common neighbour, by ZIP."""
    out = []
    for _, g in pd.DataFrame({"zip": zips, "u": u, "v": v}).groupby("zip", sort=False):
        ev = g[["u", "v"]].to_numpy()
        if len(ev) < 3:
            continue
        adj: dict[int, set[int]] = defaultdict(set)
        for a, b in ev:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
        supported = sum(1 for a, b in ev if not adj[int(a)].isdisjoint(adj[int(b)]))
        out.append(supported / len(ev))
    return np.array(out)


def economic_connectedness_by_zip(
    u: np.ndarray, v: np.ndarray, zips: np.ndarray, high_ses: pd.Series,
) -> np.ndarray:
    """Chetty-style economic connectedness for each ZIP's below-median residents.

    For every user below the sample median income, the share of their contacts
    who are above median is averaged over the ZIP's below-median residents and
    doubled, so a value of 1 means a below-median resident's contacts are, on
    average, exactly half above median -- the population-representative rate.
    """
    node_ses = high_ses.reindex(np.concatenate([u, v])).to_numpy()
    u_ses, v_ses = node_ses[: len(u)], node_ses[len(u):]

    df = pd.concat([
        pd.DataFrame({"zip": zips[u], "node": u, "ses": u_ses, "friend_ses": v_ses}),
        pd.DataFrame({"zip": zips[v], "node": v, "ses": v_ses, "friend_ses": u_ses}),
    ], ignore_index=True).dropna()
    low = df.loc[df["ses"] == False]  # noqa: E712
    if low.empty:
        return np.array([])
    return (
        low.groupby(["zip", "node"])["friend_ses"].mean().groupby(level="zip").mean() * 2
    ).to_numpy()


def run_aggregate_stage(args: argparse.Namespace) -> None:
    """Rebuild every aggregate table from the raw CDR panel."""
    voice_dir = Path(args.voice_dir).expanduser().resolve()
    sms_dir = Path(args.sms_dir).expanduser().resolve()
    for path, flag in [(voice_dir, "--voice-dir"), (sms_dir, "--sms-dir")]:
        if not path.is_dir():
            raise SystemExit(f"{flag} does not exist: {path}")

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    agg_dir = Path(args.agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)

    log("Inferring home cells from nighttime voice activity ...")
    home_cells = build_home_cells(voice_dir, Path(args.cell_map), work_dir)

    homes_rows = [
        {"threshold_events": t, "n_homes": int((home_cells["n_night_events"] >= t).sum())}
        for t in HOME_THRESHOLDS
    ]
    pd.DataFrame(homes_rows).to_csv(agg_dir / AGG_FILES["homes_by_threshold"], index=False)
    log(f"  wrote {agg_dir / AGG_FILES['homes_by_threshold']}")

    log("Mapping home cells to ZIP codes ...")
    users_zip = map_cells_to_zip(
        home_cells, Path(args.cells_csv), Path(args.zcta_shapefile),
    )
    log(f"  users with home ZIP: {len(users_zip):,}")

    log("Building the interaction network ...")
    edges_all_path = build_edges(voice_dir, sms_dir, users_zip[["msisdn", "zip"]], work_dir)
    edges_raw = pd.read_parquet(edges_all_path)
    edges_raw["n_total"] = edges_raw["n_voice"] + edges_raw["n_sms"]

    users = users_zip[["msisdn", "zip"]].drop_duplicates("msisdn").reset_index(drop=True)
    users["idx"] = np.arange(len(users))
    idx_of = users.set_index("msisdn")["idx"]
    zips_arr = users["zip"].to_numpy()
    n_nodes = len(users)

    edges_idx = edges_raw.merge(
        idx_of.rename("u_idx"), left_on="u", right_index=True
    ).merge(idx_of.rename("v_idx"), left_on="v", right_index=True)
    edges_idx = edges_idx.loc[edges_idx["u_idx"] != edges_idx["v_idx"]]

    edge_thresholds = sorted(set(EDGE_THRESHOLDS) | {MIN_EDGE_INTERACTIONS})
    edges_rows, degree_rows, strength_rows = [], [], []
    baseline_degree = None
    for thresh in edge_thresholds:
        sub = edges_idx.loc[edges_idx["n_total"] >= thresh]
        nodes = np.concatenate([sub["u_idx"].to_numpy(), sub["v_idx"].to_numpy()])
        deg = pd.Series(nodes).value_counts().to_numpy()
        edges_rows.append({"threshold_m": thresh, "n_edges": len(sub), "n_nodes": len(np.unique(nodes))})

        k_vals = np.arange(1, int(deg.max()) + 1) if len(deg) else np.array([])
        counts = np.bincount(deg, minlength=int(deg.max()) + 1)[1:] if len(deg) else np.array([])
        strength = counts / counts.sum() if counts.sum() else counts
        for k, p in zip(k_vals, strength):
            if p > 0:
                strength_rows.append({"threshold_m": thresh, "degree": int(k), "p": float(p)})

        if thresh == MIN_EDGE_INTERACTIONS:
            baseline_degree = deg
    pd.DataFrame(edges_rows).to_csv(agg_dir / AGG_FILES["edges_by_threshold"], index=False)
    pd.DataFrame(strength_rows).to_csv(agg_dir / AGG_FILES["strength_by_threshold"], index=False)
    log(f"  wrote {agg_dir / AGG_FILES['edges_by_threshold']}")
    log(f"  wrote {agg_dir / AGG_FILES['strength_by_threshold']}")

    deg_bins = np.arange(0, 101, 1)
    counts, edges_hist = np.histogram(baseline_degree, bins=deg_bins, density=True)
    degree_dist = pd.DataFrame({
        "degree": 0.5 * (edges_hist[1:] + edges_hist[:-1]), "density": counts,
    })
    degree_dist.to_csv(agg_dir / AGG_FILES["degree_dist"], index=False)
    log(f"  wrote {agg_dir / AGG_FILES['degree_dist']}")

    log("Counting unique locations visited per user ...")
    node_msisdn = users["msisdn"].to_numpy()
    unique_locs = build_unique_locations(voice_dir, Path(args.cell_map), node_msisdn)
    s_bins = np.arange(0, 101, 2)
    counts, edges_hist = np.histogram(unique_locs, bins=s_bins, density=True)
    unique_dist = pd.DataFrame({
        "locations": 0.5 * (edges_hist[1:] + edges_hist[:-1]), "density": counts,
    })
    unique_dist.to_csv(agg_dir / AGG_FILES["unique_locations_dist"], index=False)
    log(f"  wrote {agg_dir / AGG_FILES['unique_locations_dist']}")

    log("Computing ZIP-level expansion factors ...")
    n_users_cdr = users.groupby("zip", as_index=False).size().rename(
        columns={"size": "n_users_cdr"}
    )
    pop = pd.read_csv(args.population_lookup, dtype={"zip": str})
    pop["zip"] = zfill_zip(pop["zip"])
    ef = n_users_cdr.merge(pop[["zip", "population"]], on="zip", how="inner")
    ef["expansion_factor"] = ef["population"] / ef["n_users_cdr"]
    ef = ef.replace([np.inf, -np.inf], np.nan).dropna(subset=["expansion_factor"])
    ef[["zip", "expansion_factor"]].to_csv(
        agg_dir / AGG_FILES["expansion_factor_zip"], index=False
    )
    log(f"  wrote {agg_dir / AGG_FILES['expansion_factor_zip']}")

    grid_ef = kde_on_grid(
        ef["expansion_factor"].to_numpy(),
        (0.0, float(np.percentile(ef["expansion_factor"], 99))),
        bw_method="scott",
    )
    if grid_ef is not None:
        pd.DataFrame({"expansion_factor": grid_ef[0], "density": grid_ef[1]}).to_csv(
            agg_dir / AGG_FILES["expansion_factor_kde"], index=False
        )
        log(f"  wrote {agg_dir / AGG_FILES['expansion_factor_kde']}")

    log("Computing social capital metrics across interaction thresholds ...")
    high_ses = None
    if args.income_lookup:
        income = pd.read_csv(args.income_lookup, dtype={"zip": str})
        income["zip"] = zfill_zip(income["zip"])
        node_income = pd.Series(zips_arr, index=np.arange(n_nodes)).map(
            income.set_index("zip")["median_income"]
        )
        high_ses = node_income > node_income.median()
    else:
        log("  --income-lookup not given; economic connectedness will be skipped")

    kde_rows = []
    for thresh in EDGE_THRESHOLDS:
        sub = edges_idx.loc[edges_idx["n_total"] >= thresh]
        u_arr = sub["u_idx"].to_numpy(dtype=np.int64)
        v_arr = sub["v_idx"].to_numpy(dtype=np.int64)
        within = zips_arr[u_arr] == zips_arr[v_arr]

        clust = clustering_by_zip(u_arr, v_arr, n_nodes, zips_arr)
        sr = support_ratio_by_zip(u_arr[within], v_arr[within], zips_arr[u_arr[within]])
        ec = (
            economic_connectedness_by_zip(u_arr, v_arr, zips_arr, high_ses)
            if high_ses is not None else np.array([])
        )
        log(f"  m={thresh}: clustering={np.nanmean(clust):.3f}  support_ratio={np.nanmean(sr):.3f}"
            + (f"  EC={np.nanmean(ec):.3f}" if len(ec) else ""))

        for key, values in [("clust", clust), ("sr", sr), ("ec", ec)]:
            xlim = dict((k, x) for k, _, x in SC_PANELS)[key]
            grid = kde_on_grid(values, xlim)
            if grid is None:
                continue
            kde_rows.append(pd.DataFrame({
                "threshold_m": thresh, "metric": key, "x": grid[0], "density": grid[1],
            }))
    if kde_rows:
        pd.concat(kde_rows, ignore_index=True).to_csv(
            agg_dir / AGG_FILES["soccap_kde"], index=False
        )
        log(f"  wrote {agg_dir / AGG_FILES['soccap_kde']}")


# --------------------------------------------------------------------------
# Stage 2: figures from the shared aggregates
# --------------------------------------------------------------------------

def plot_figure_s1(agg_dir: Path, plots_dir: Path, dpi: int) -> None:
    """Figure S1: network construction and CDR-vs-census sanity checks."""
    edges = read_aggregate(agg_dir, "edges_by_threshold")
    degree_dist = read_aggregate(agg_dir, "degree_dist")
    ef_kde = read_aggregate(agg_dir, "expansion_factor_kde")
    ef_zip = read_aggregate(agg_dir, "expansion_factor_zip")
    homes = read_aggregate(agg_dir, "homes_by_threshold")
    unique_dist = read_aggregate(agg_dir, "unique_locations_dist")

    require_columns(edges, ["threshold_m", "n_edges"], AGG_FILES["edges_by_threshold"])
    require_columns(degree_dist, ["degree", "density"], AGG_FILES["degree_dist"])
    require_columns(homes, ["threshold_events", "n_homes"], AGG_FILES["homes_by_threshold"])
    require_columns(unique_dist, ["locations", "density"], AGG_FILES["unique_locations_dist"])

    plt.rcParams.update({
        "font.size": FS, "axes.labelsize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS,
    })
    fig = plt.figure(figsize=(22, 13), dpi=dpi)
    gs = fig.add_gridspec(2, 3, wspace=0.32, hspace=0.35)
    ax_edges = fig.add_subplot(gs[0, 0])
    ax_deg = fig.add_subplot(gs[0, 1])
    ax_exp = fig.add_subplot(gs[0, 2])
    ax_homes = fig.add_subplot(gs[1, 0])
    ax_locs = fig.add_subplot(gs[1, 1])
    ax_map = fig.add_subplot(gs[1, 2])

    def apply_ticks(ax, n=3):
        ax.xaxis.set_major_locator(mticker.MaxNLocator(n))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(n))
        ax.tick_params(labelsize=FS)

    # A: edges against the interaction threshold.
    edges = edges.sort_values("threshold_m")
    ax_edges.plot(edges["threshold_m"], edges["n_edges"] / 1e6, "o-", lw=2, color="tab:blue")
    ax_edges.axvline(MIN_EDGE_INTERACTIONS, color="tab:red", ls="--", lw=1.8,
                     label=f"$m={MIN_EDGE_INTERACTIONS}$")
    ax_edges.set_xlabel("Interaction threshold $m$", fontsize=FS)
    ax_edges.set_ylabel("Edges (millions)", fontsize=FS)
    ax_edges.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    apply_ticks(ax_edges)
    ax_edges.legend(fontsize=FS - 2)
    ax_edges.text(-0.15, 1.05, "A", transform=ax_edges.transAxes, fontsize=FS,
                 fontweight="bold", va="top")

    # B: degree distribution at the baseline threshold.
    degree_dist = degree_dist.sort_values("degree")
    ax_deg.plot(degree_dist["degree"], degree_dist["density"], marker="o", lw=2, ms=4,
               color="tab:blue", label="$p(k)$")
    ax_deg.set_xlim(0, 100)
    ax_deg.set_xticks([0, 30, 60, 90])
    ax_deg.set_xlabel("Degree $k$", fontsize=FS)
    ax_deg.set_ylabel("Density", fontsize=FS)
    ax_deg.tick_params(labelsize=FS)
    ax_deg.legend(fontsize=FS - 2)
    ax_deg.text(-0.15, 1.05, "B", transform=ax_deg.transAxes, fontsize=FS,
               fontweight="bold", va="top")

    # C: distribution of the ZIP-level expansion factor.
    if not ef_kde.empty:
        require_columns(ef_kde, ["expansion_factor", "density"], AGG_FILES["expansion_factor_kde"])
        ax_exp.plot(ef_kde["expansion_factor"], ef_kde["density"], lw=2.2, color="tab:purple")
    ax_exp.set_xlabel("Expansion factor", fontsize=FS)
    ax_exp.set_ylabel("Density", fontsize=FS)
    apply_ticks(ax_exp)
    ax_exp.text(-0.15, 1.05, "C", transform=ax_exp.transAxes, fontsize=FS,
               fontweight="bold", va="top")

    # D: inferred homes against the night-events threshold.
    homes = homes.sort_values("threshold_events")
    ax_homes.plot(homes["threshold_events"], homes["n_homes"] / 1000, "o-", lw=2.2,
                 color="tab:blue", ms=7)
    ax_homes.axvline(MIN_HOME_NIGHT_EVENTS, color="tab:red", ls="--", lw=1.8,
                     label=f"$m={MIN_HOME_NIGHT_EVENTS}$")
    for _, row in homes.iterrows():
        t = row["threshold_events"]
        if t in (homes["threshold_events"].iloc[0], MIN_HOME_NIGHT_EVENTS, homes["threshold_events"].iloc[-1]):
            ax_homes.annotate(f"{row['n_homes'] / 1000:.0f}k", xy=(t, row["n_homes"] / 1000),
                              xytext=(4, 4), textcoords="offset points", fontsize=FS - 4)
    ax_homes.set_xlabel("Night-events threshold $m$", fontsize=FS)
    ax_homes.set_ylabel("Inferred homes (thousands)", fontsize=FS)
    apply_ticks(ax_homes)
    ax_homes.legend(fontsize=FS - 2)
    ax_homes.text(-0.15, 1.05, "D", transform=ax_homes.transAxes, fontsize=FS,
                 fontweight="bold", va="top")

    # E: distribution of unique locations visited.
    unique_dist = unique_dist.sort_values("locations")
    ax_locs.plot(unique_dist["locations"], unique_dist["density"], marker="s", lw=2, ms=4,
                color="tab:red", label="$p(S)$")
    ax_locs.set_xlim(0, 100)
    ax_locs.set_xlabel("Number of Unique Locations", fontsize=FS)
    ax_locs.set_ylabel("Density", fontsize=FS)
    ax_locs.tick_params(labelsize=FS)
    ax_locs.legend(fontsize=FS - 2)
    ax_locs.text(-0.15, 1.05, "E", transform=ax_locs.transAxes, fontsize=FS,
                fontweight="bold", va="top")

    # F: choropleth of the expansion factor across the Bay Area.
    if not ef_zip.empty:
        require_columns(ef_zip, ["zip", "expansion_factor"], AGG_FILES["expansion_factor_zip"])
        plot_expansion_factor_map(ax_map, ef_zip, plots_dir)
    else:
        ax_map.axis("off")
        log("  expansion factor table unavailable; map panel omitted")
    ax_map.text(-0.05, 1.05, "F", transform=ax_map.transAxes, fontsize=FS,
               fontweight="bold", va="top")

    out = plots_dir / "figure_s1.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def plot_expansion_factor_map(ax, ef_zip: pd.DataFrame, plots_dir: Path) -> None:
    """Choropleth of the ZIP-level expansion factor, restricted to the Bay Area."""
    import geopandas as gpd
    import matplotlib.colors as mcolors
    from shapely.geometry import box as sbox

    boundary_path = plots_dir.parent / "data" / "processed" / "cdr_aggregates" / "bay_area_zctas.gpkg"
    if not boundary_path.exists():
        ax.axis("off")
        log(f"  ZIP boundaries not found at {boundary_path}; map panel omitted")
        return

    gdf = gpd.read_file(boundary_path)
    gdf["zip"] = zfill_zip(gdf["zip"])
    merged = gdf.merge(ef_zip, on="zip", how="inner").to_crs("EPSG:3857")

    bay_box = gpd.GeoSeries(
        [sbox(BAY_AREA_BBOX["lon_min"], BAY_AREA_BBOX["lat_min"],
             BAY_AREA_BBOX["lon_max"], BAY_AREA_BBOX["lat_max"])],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857").iloc[0]
    merged = merged[merged.geometry.intersects(bay_box)]
    if merged.empty:
        ax.axis("off")
        return

    vmin = np.percentile(merged["expansion_factor"], 2)
    vmax = np.percentile(merged["expansion_factor"], 98)
    merged.plot(column="expansion_factor", ax=ax, cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
               legend=False, edgecolor="white", linewidth=0.3)
    try:
        import contextily as ctx

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik,
                        attribution=False, crs="EPSG:3857", alpha=0.5)
    except Exception:
        ax.set_facecolor("#eaeaea")

    ax.set_xlim(bay_box.bounds[0], bay_box.bounds[2])
    ax.set_ylim(bay_box.bounds[1], bay_box.bounds[3])
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Expansion factor", fontsize=FS - 6)
    cb.ax.tick_params(labelsize=FS - 6)
    cb.locator = mticker.MaxNLocator(3)
    cb.update_ticks()


def plot_figure_s2(agg_dir: Path, plots_dir: Path, dpi: int) -> None:
    """Figure S2: sensitivity of degree and social capital metrics to m."""
    strength = read_aggregate(agg_dir, "strength_by_threshold")
    kde = read_aggregate(agg_dir, "soccap_kde")
    require_columns(strength, ["threshold_m", "degree", "p"], AGG_FILES["strength_by_threshold"])
    require_columns(kde, ["threshold_m", "metric", "x", "density"], AGG_FILES["soccap_kde"])

    thresholds = sorted(set(strength["threshold_m"]) | set(kde["threshold_m"]))
    edge_colors = plt.cm.plasma_r(np.linspace(0.12, 0.88, len(thresholds)))
    sc_colors = plt.cm.Greens(np.linspace(0.25, 0.92, len(thresholds)))

    plt.rcParams.update({"font.size": FS, "axes.labelsize": FS,
                         "xtick.labelsize": FS, "ytick.labelsize": FS})
    fig, axes = plt.subplots(1, 4, figsize=(26, 7), dpi=dpi)
    ax_strength, ax_clust, ax_sr, ax_ec = axes

    for thresh, color in zip(thresholds, edge_colors):
        sub = strength.loc[strength["threshold_m"] == thresh].sort_values("degree")
        if sub.empty:
            continue
        lw = 2.5 if thresh == MIN_EDGE_INTERACTIONS else 1.5
        ls = "-" if thresh == MIN_EDGE_INTERACTIONS else "--"
        ax_strength.plot(sub["degree"], sub["p"], lw=lw, ls=ls, color=color, label=f"$m={thresh}$")
    ax_strength.set_xscale("log")
    ax_strength.set_yscale("log")
    ax_strength.set_xlabel("Strength $s$", fontsize=FS)
    ax_strength.set_ylabel("$p(s)$", fontsize=FS)
    ax_strength.tick_params(labelsize=FS)
    ax_strength.legend(ncol=2, framealpha=0.9, fontsize=FS - 6)
    ax_strength.text(-0.15, 1.05, "A", transform=ax_strength.transAxes, fontsize=FS,
                     fontweight="bold", va="top")

    panel_axes = {"clust": ax_clust, "sr": ax_sr, "ec": ax_ec}
    panel_labels = {"clust": ("B", "Clustering"), "sr": ("C", "Support Ratio"),
                    "ec": ("D", "Economic Connectedness")}
    for key, ax in panel_axes.items():
        label, xlabel = panel_labels[key]
        sub_metric = kde.loc[kde["metric"] == key]
        if sub_metric.empty:
            ax.axis("off")
            continue
        for thresh, color in zip(thresholds, sc_colors):
            sub = sub_metric.loc[sub_metric["threshold_m"] == thresh].sort_values("x")
            if sub.empty:
                continue
            lw = 2.2 if thresh == MIN_EDGE_INTERACTIONS else 1.2
            ax.fill_between(sub["x"], sub["density"], color=color, alpha=0.30)
            ax.plot(sub["x"], sub["density"], color=color, lw=lw)
        ax.set_xlabel(xlabel, fontsize=FS)
        ax.set_ylabel("Density", fontsize=FS)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", ls="--", alpha=0.4)
        ax.tick_params(labelsize=FS)
        ax.text(-0.15, 1.05, label, transform=ax.transAxes, fontsize=FS,
               fontweight="bold", va="top")

    sm = plt.cm.ScalarMappable(cmap="Greens",
                                norm=plt.Normalize(vmin=thresholds[0], vmax=thresholds[-1]))
    sm.set_array([])
    cax = ax_clust.inset_axes([0.72, 0.2, 0.05, 0.6])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Threshold $m$", fontsize=FS - 4)
    cb.ax.tick_params(labelsize=FS - 4)
    cb.locator = mticker.MaxNLocator(3)
    cb.update_ticks()

    plt.tight_layout()
    out = plots_dir / "figure_s2.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def run_figures_stage(args: argparse.Namespace) -> None:
    """Render the requested figures from the shared aggregates."""
    agg_dir = Path(args.agg_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    renderers = {"s1": plot_figure_s1, "s2": plot_figure_s2}
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
             "'aggregate' rebuilds those aggregates from the raw CDR panel.",
    )
    parser.add_argument(
        "--figures", nargs="+", choices=["s1", "s2"], default=["s1", "s2"],
        help="Subset of figures to produce (default: all).",
    )
    parser.add_argument(
        "--data-dir", default=str(here / "data"),
        help="Repository data directory (default: %(default)s).",
    )
    parser.add_argument(
        "--agg-dir", default=None,
        help="Directory holding the CDR aggregate tables "
             "(default: <data-dir>/processed/cdr_aggregates).",
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
        "--voice-dir",
        help="Directory of daily voice call parquet files. The raw CDR data is "
             "proprietary and is not distributed with this repository.",
    )
    raw.add_argument(
        "--sms-dir",
        help="Directory of daily SMS parquet files.",
    )
    raw.add_argument(
        "--cell-map",
        help="LZMA pickle mapping raw carrier cell ids to normalised cell ids.",
    )
    raw.add_argument(
        "--cells-csv",
        help="CSV of cell_id, latitude, longitude for every cell tower.",
    )
    raw.add_argument(
        "--zcta-shapefile",
        help="TIGER/Line ZCTA shapefile, used to map home cells to ZIP codes.",
    )
    raw.add_argument(
        "--population-lookup",
        help="CSV of zip and population (ACS table B01003_001E), used for the "
             "expansion factor.",
    )
    raw.add_argument(
        "--income-lookup", default=None,
        help="CSV of zip and median_income, used to split users for economic "
             "connectedness. Optional: EC is skipped without it.",
    )
    raw.add_argument(
        "--work-dir", default=None,
        help="Scratch directory for intermediate parquet files "
             "(default: <agg-dir>/../cdr_work).",
    )

    args = parser.parse_args(argv)
    if args.agg_dir is None:
        args.agg_dir = str(Path(args.data_dir) / "processed" / "cdr_aggregates")
    if args.work_dir is None:
        args.work_dir = str(Path(args.agg_dir) / ".." / "cdr_work")

    if args.stage == "aggregate":
        missing = [
            name for name, value in [
                ("--voice-dir", args.voice_dir), ("--sms-dir", args.sms_dir),
                ("--cell-map", args.cell_map), ("--cells-csv", args.cells_csv),
                ("--zcta-shapefile", args.zcta_shapefile),
                ("--population-lookup", args.population_lookup),
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
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
