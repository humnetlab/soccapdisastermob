#!/usr/bin/env python3
"""
02_compare_soccap_metrics.py -- comparison of social capital metrics.

    figure_2.png   CDR-derived support ratio and clustering against the Social
                   Capital Atlas equivalents, at ZIP level: a hexbin density of
                   the two measurements, and the two distributions side by
                   side, for each metric.
    figure_s3.png  Friendship probability against home distance, for the CDR
                   network and for Meta's Social Connectedness Index (SCI),
                   both nationwide and restricted to the Bay Area, each
                   relative to a geographic null of random user pairs.

Both figures compare the CDR-derived social network from
``01_processing_cdr.py`` (m=3) against Meta-derived social capital measures at
ZIP level. Figure 2 checks the two ZIP-level metrics that both sources
publish, support ratio and clustering; figure S3 checks the shape of the
underlying network rather than any one summary statistic, by comparing how
sharply tie probability falls off with distance.

The raw traces cannot be redistributed, so the script runs in two stages:
``--stage figures`` (default) draws from the shared aggregates in ``data/`` and
needs no raw data; ``--stage aggregate`` rebuilds them from the CDR network
built by ``01_processing_cdr.py --stage aggregate`` and from Meta's Social
Connectedness Index. Run with --help for options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------
# Study configuration
# --------------------------------------------------------------------------

#: Interaction threshold the comparison is run at, matching the main CDR analysis.
MIN_EDGE_INTERACTIONS = 3

#: ZIP codes need at least this many within-ZIP edges to enter figure 2, so a
#: handful of interactions cannot dominate a ZIP's estimate.
MIN_EDGES_FOR_COMPARISON = 50

#: Metrics compared in figure 2, and the Social Capital Atlas column each maps to.
METRICS = [
    ("support_ratio", "support_ratio_zip", (0, 1)),
    ("clustering", "clustering_zip", (0, 0.7)),
]

#: Bay Area bounding box, matching ``01_processing_cdr.py``.
BAY_AREA_BBOX = {"lat_min": 37.0, "lat_max": 38.3, "lon_min": -123.0, "lon_max": -121.5}

#: log-spaced distance bins (km) for figure S3, and the random-pair sample size
#: used to build the geographic null.
DIST_BINS_KM = np.logspace(np.log10(0.5), np.log10(500), 55)
N_RANDOM_PAIRS = 1_000_000
RANDOM_SEED = 42
#: A distance bin needs at least this many SCI ZIP pairs to be plotted.
MIN_SCI_PAIRS_PER_BIN = {"all": 10, "bay_area": 5}
#: A distance bin needs at least this many CDR edges to be plotted.
MIN_CDR_PAIRS_PER_BIN = 50

FS = 13

#: Aggregate tables written by ``--stage aggregate``.
AGG_FILES = {
    "zip_metrics": "zip_cdr_vs_facebook.csv",
    "distance_decay": "distance_decay.csv",
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


def read_aggregate(agg_dir: Path, key: str) -> pd.DataFrame:
    """Load one aggregate table, with a message pointing at the other stage."""
    path = agg_dir / AGG_FILES[key]
    if not path.exists():
        raise SystemExit(
            f"Missing aggregate table: {path}\n"
            f"Run `python {Path(__file__).name} --stage aggregate` with access to "
            f"the CDR network and Social Connectedness Index to regenerate it, "
            f"or restore the file shipped with the repository."
        )
    return pd.read_csv(path, dtype={"zip": str})


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance between two arrays of (lat, lon) points, in km."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def pair_density_ratio(
    weights: np.ndarray, dist: np.ndarray, bins: np.ndarray, min_count: int,
) -> np.ndarray:
    """Weighted share of pairs at distance d, relative to the share of all pairs.

    A value above 1 at distance d means pairs at that distance are
    over-represented relative to how common that distance is among all pairs
    in the same set, i.e. relative to a geographic null.
    """
    n = len(bins) - 1
    idx = np.searchsorted(bins, dist, side="right") - 1
    valid = (idx >= 0) & (idx < n)
    weight_sums = np.zeros(n)
    pair_counts = np.zeros(n)
    np.add.at(weight_sums, idx[valid], weights[valid])
    np.add.at(pair_counts, idx[valid], 1)
    weight_density = weight_sums / weight_sums.sum()
    pair_density = pair_counts / pair_counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pair_counts >= min_count, weight_density / pair_density, np.nan)
    return ratio


# --------------------------------------------------------------------------
# Stage 1: aggregation
# --------------------------------------------------------------------------

def build_zip_metrics_table(
    edges_all_path: Path, users_zip_path: Path, social_capital_path: Path,
) -> pd.DataFrame:
    """CDR support ratio and clustering per ZIP, alongside the Atlas values.

    Recomputes both CDR metrics at the main interaction threshold from the
    network built by ``01_processing_cdr.py --stage aggregate``, so this
    script does not need its own copy of the metric logic.
    """
    import importlib.util
    from collections import defaultdict

    # Reuse the metric functions from 01_processing_cdr.py rather than
    # duplicating them; imported by file path since the module name starts
    # with a digit and cannot be imported normally.
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "processing_cdr", here / "01_processing_cdr.py"
    )
    cdr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cdr)

    edges_raw = pd.read_parquet(edges_all_path)
    edges_raw["n_total"] = edges_raw["n_voice"] + edges_raw["n_sms"]
    edges_raw = edges_raw.loc[edges_raw["n_total"] >= MIN_EDGE_INTERACTIONS]

    users = pd.read_parquet(users_zip_path)[["msisdn", "zip"]].drop_duplicates("msisdn")
    users["zip"] = zfill_zip(users["zip"])
    users = users.reset_index(drop=True)
    users["idx"] = np.arange(len(users))
    idx_of = users.set_index("msisdn")["idx"]
    zips_arr = users["zip"].to_numpy()
    n_nodes = len(users)

    edges_idx = edges_raw.merge(
        idx_of.rename("u_idx"), left_on="u", right_index=True
    ).merge(idx_of.rename("v_idx"), left_on="v", right_index=True)
    edges_idx = edges_idx.loc[edges_idx["u_idx"] != edges_idx["v_idx"]]
    u_arr = edges_idx["u_idx"].to_numpy(dtype=np.int64)
    v_arr = edges_idx["v_idx"].to_numpy(dtype=np.int64)
    within = zips_arr[u_arr] == zips_arr[v_arr]

    clust = cdr.clustering_by_zip(u_arr, v_arr, n_nodes, zips_arr)
    clust_zip = pd.DataFrame({"zip": zips_arr, "clust": cdr.local_clustering(u_arr, v_arr, n_nodes)})
    clust_by_zip = clust_zip.groupby("zip", as_index=False).agg(
        clustering_cdr=("clust", "mean"), n_users=("clust", "size"),
    )

    sr_rows = []
    for zip_code, g in pd.DataFrame({
        "zip": zips_arr[u_arr[within]], "u": u_arr[within], "v": v_arr[within],
    }).groupby("zip", sort=False):
        ev = g[["u", "v"]].to_numpy()
        adj: dict[int, set[int]] = defaultdict(set)
        for a, b in ev:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
        supported = sum(1 for a, b in ev if not adj[int(a)].isdisjoint(adj[int(b)]))
        sr_rows.append({"zip": zip_code, "n_edges": len(ev), "support_ratio_cdr": supported / len(ev)})
    sr_by_zip = pd.DataFrame(sr_rows)

    cdr_zip = sr_by_zip.merge(clust_by_zip, on="zip", how="outer")

    fb = pd.read_csv(social_capital_path, dtype={"zip": str}, low_memory=False)
    fb["zip"] = zfill_zip(fb["zip"])
    fb = fb[["zip", "support_ratio_zip", "clustering_zip"]].rename(
        columns={"support_ratio_zip": "support_ratio_fb", "clustering_zip": "clustering_fb"}
    )

    return cdr_zip.merge(fb, on="zip", how="inner")


def build_distance_decay_table(
    edges_all_path: Path, users_zip_path: Path, cells_csv: Path,
    zcta_shapefile: Path, sci_glob: str, duckdb_threads: int,
) -> pd.DataFrame:
    """Friendship probability against distance, for CDR and for Meta's SCI.

    Both sources are expressed relative to a geographic null so the two are
    comparable despite different units: for the CDR network, random pairs of
    users; for SCI, the distribution of distances between all ZIP pairs
    carrying a nonzero score. Within-ZIP pairs use the square root of the ZIP
    code's area rather than a zero centroid-to-centroid distance.
    """
    import duckdb
    import geopandas as gpd

    zcta = gpd.read_file(zcta_shapefile)
    zip_col = next(
        (c for c in ["ZCTA5CE20", "ZCTA5CE10", "ZCTA5CE"] if c in zcta.columns),
        next(c for c in zcta.columns if "ZCTA5" in c.upper()),
    )
    zcta = zcta[[zip_col, "geometry"]].rename(columns={zip_col: "zip"})
    zcta["zip"] = zfill_zip(zcta["zip"])
    if zcta.crs is not None and zcta.crs.to_epsg() != 4326:
        zcta = zcta.to_crs(epsg=4326)
    zcta["area_km2"] = zcta.to_crs("EPSG:3857").geometry.area / 1e6
    zcta["lat"] = zcta.geometry.centroid.y
    zcta["lon"] = zcta.geometry.centroid.x
    centroids = zcta[["zip", "lat", "lon", "area_km2"]].drop_duplicates("zip").set_index("zip")

    bay_mask = (
        centroids["lat"].between(BAY_AREA_BBOX["lat_min"], BAY_AREA_BBOX["lat_max"])
        & centroids["lon"].between(BAY_AREA_BBOX["lon_min"], BAY_AREA_BBOX["lon_max"])
    )
    bay_zips = set(centroids.index[bay_mask])
    zip_area = centroids["area_km2"]

    # --- CDR: friendship probability against a random-pair geographic null. ---
    log("  computing CDR distance decay ...")
    cells_df = pd.read_csv(cells_csv, usecols=["cell_id", "latitude", "longitude"])
    home = pd.read_parquet(users_zip_path)
    home = home.rename(columns={"A_MSISDN": "msisdn"}) if "A_MSISDN" in home.columns else home
    home["zip"] = zfill_zip(home["zip"])
    home = (
        home[["msisdn", "home_cell", "zip"]].drop_duplicates("msisdn")
        .merge(cells_df.rename(columns={"cell_id": "home_cell"}), on="home_cell", how="inner")
        .rename(columns={"latitude": "lat", "longitude": "lon"})
    )

    edges = pd.read_parquet(edges_all_path)
    edges["n_total"] = edges["n_voice"] + edges["n_sms"]
    edges = edges.loc[edges["n_total"] >= MIN_EDGE_INTERACTIONS, ["u", "v"]].drop_duplicates()
    loc = home.set_index("msisdn")
    edges = edges.loc[edges["u"].isin(loc.index) & edges["v"].isin(loc.index)].copy()
    edges["lat_u"] = loc.loc[edges["u"].to_numpy(), "lat"].to_numpy()
    edges["lon_u"] = loc.loc[edges["u"].to_numpy(), "lon"].to_numpy()
    edges["lat_v"] = loc.loc[edges["v"].to_numpy(), "lat"].to_numpy()
    edges["lon_v"] = loc.loc[edges["v"].to_numpy(), "lon"].to_numpy()
    edges["zip_u"] = loc.loc[edges["u"].to_numpy(), "zip"].to_numpy()
    edges["zip_v"] = loc.loc[edges["v"].to_numpy(), "zip"].to_numpy()

    edge_dists = haversine_km(
        edges["lat_u"].to_numpy(), edges["lon_u"].to_numpy(),
        edges["lat_v"].to_numpy(), edges["lon_v"].to_numpy(),
    )
    within_mask = (edges["zip_u"] == edges["zip_v"]).to_numpy()
    edge_dists[within_mask] = edges.loc[within_mask, "zip_u"].map(zip_area).apply(np.sqrt).to_numpy()

    rng = np.random.default_rng(RANDOM_SEED)
    coords = home[["lat", "lon"]].to_numpy()
    idx = rng.integers(0, len(coords), size=(N_RANDOM_PAIRS, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    rand_dists = haversine_km(
        coords[idx[:, 0], 0], coords[idx[:, 0], 1], coords[idx[:, 1], 0], coords[idx[:, 1], 1],
    )

    bins = DIST_BINS_KM
    fc, _ = np.histogram(edge_dists, bins=bins)
    nc, _ = np.histogram(rand_dists, bins=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_d_cdr = np.where(nc > 0, (fc / fc.sum()) / (nc / nc.sum()), np.nan)
    p_d_cdr = np.where(nc >= MIN_CDR_PAIRS_PER_BIN, p_d_cdr, np.nan)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    # --- Meta SCI: same ratio, computed on ZIP-pair SCI weights. ---
    log("  computing SCI distance decay ...")
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={duckdb_threads}")
    sci = con.execute(f"""
        SELECT LPAD(CAST(user_region AS VARCHAR), 5, '0') AS z1,
               LPAD(CAST(friend_region AS VARCHAR), 5, '0') AS z2,
               scaled_sci::DOUBLE AS sci
        FROM read_csv_auto('{sci_glob}')
        WHERE user_country = 'US' AND friend_country = 'US'
    """).df()
    con.close()

    sci = sci.loc[sci["z1"].isin(centroids.index) & sci["z2"].isin(centroids.index)].copy()
    sci["lat1"] = centroids.loc[sci["z1"].to_numpy(), "lat"].to_numpy()
    sci["lon1"] = centroids.loc[sci["z1"].to_numpy(), "lon"].to_numpy()
    sci["lat2"] = centroids.loc[sci["z2"].to_numpy(), "lat"].to_numpy()
    sci["lon2"] = centroids.loc[sci["z2"].to_numpy(), "lon"].to_numpy()
    sci["dist_km"] = haversine_km(
        sci["lat1"].to_numpy(), sci["lon1"].to_numpy(), sci["lat2"].to_numpy(), sci["lon2"].to_numpy(),
    )
    same_mask = (sci["z1"] == sci["z2"]).to_numpy()
    sci.loc[same_mask, "dist_km"] = sci.loc[same_mask, "z1"].map(zip_area).apply(np.sqrt).to_numpy()

    sci_ba = sci.loc[sci["z1"].isin(bay_zips) & sci["z2"].isin(bay_zips)]
    p_d_sci_all = pair_density_ratio(
        sci["sci"].to_numpy(), sci["dist_km"].to_numpy(), bins, MIN_SCI_PAIRS_PER_BIN["all"]
    )
    p_d_sci_ba = pair_density_ratio(
        sci_ba["sci"].to_numpy(), sci_ba["dist_km"].to_numpy(), bins, MIN_SCI_PAIRS_PER_BIN["bay_area"]
    )

    return pd.DataFrame({
        "distance_km": bin_centers,
        "p_cdr_bay_area": p_d_cdr,
        "p_sci_us": p_d_sci_all,
        "p_sci_bay_area": p_d_sci_ba,
    })


def run_aggregate_stage(args: argparse.Namespace) -> None:
    """Rebuild both aggregate tables from the CDR network and the SCI."""
    agg_dir = Path(args.agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)

    log("Comparing CDR and Facebook ZIP-level metrics ...")
    zip_metrics = build_zip_metrics_table(
        Path(args.edges_all), Path(args.users_zip), Path(args.social_capital),
    )
    zip_metrics.to_csv(agg_dir / AGG_FILES["zip_metrics"], index=False)
    log(f"  wrote {agg_dir / AGG_FILES['zip_metrics']}")

    if args.sci_glob:
        log("Computing distance decay for CDR and Meta SCI ...")
        decay = build_distance_decay_table(
            Path(args.edges_all), Path(args.users_zip), Path(args.cells_csv),
            Path(args.zcta_shapefile), args.sci_glob, args.duckdb_threads,
        )
        decay.to_csv(agg_dir / AGG_FILES["distance_decay"], index=False)
        log(f"  wrote {agg_dir / AGG_FILES['distance_decay']}")
    else:
        log("  --sci-glob not given; skipping figure S3's aggregate")


# --------------------------------------------------------------------------
# Stage 2: figures
# --------------------------------------------------------------------------

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def plot_figure_2(agg_dir: Path, plots_dir: Path, dpi: int) -> None:
    """Figure 2: CDR vs. Facebook support ratio and clustering, at ZIP level."""
    data = read_aggregate(agg_dir, "zip_metrics")
    require_columns(
        data,
        ["zip", "support_ratio_cdr", "clustering_cdr", "support_ratio_fb", "clustering_fb", "n_edges"],
        AGG_FILES["zip_metrics"],
    )

    stable = data.loc[data["n_edges"].fillna(0) >= MIN_EDGES_FOR_COMPARISON]

    plt.rcParams.update({
        "figure.dpi": dpi, "savefig.dpi": dpi,
        "axes.titlesize": 18, "axes.labelsize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=dpi)

    for row, (key, _fb_column, xlim) in enumerate(METRICS):
        cdr_col = f"{key}_cdr"
        fb_col = f"{key}_fb"

        x = stable[cdr_col].to_numpy()
        y = stable[fb_col].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)

        ax = axes[row, 0]
        hb = ax.hexbin(x[mask], y[mask], gridsize=40, mincnt=1, cmap="viridis")
        lo, hi = xlim
        ax.plot([lo, hi], [lo, hi], color="white", lw=2, alpha=0.9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        rho = pearson(x, y)
        ax.text(0.03, 0.97, rf"Pearson $\rho$ = {rho:.3f}" + f"\nN = {int(mask.sum())}",
               transform=ax.transAxes, va="top", ha="left", fontsize=12,
               bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, boxstyle="round,pad=0.3"))
        ax.set_xlabel(f"CDR {key.replace('_', ' ')}")
        ax.set_ylabel(f"Facebook {key.replace('_', ' ')}")
        fig.colorbar(hb, ax=ax).set_label("ZIP count")

        ax = axes[row, 1]
        bins = np.linspace(lo, hi, 41)
        ax.hist(x[mask], bins=bins, alpha=0.55, density=True, label="CDR")
        ax.hist(y[mask], bins=bins, alpha=0.55, density=True, label="Facebook")
        ax.set_xlabel(key.replace("_", " "))
        ax.set_ylabel("Density")
        ax.legend(frameon=True)

    plt.tight_layout()
    out = plots_dir / "figure_2.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def plot_figure_s3(agg_dir: Path, plots_dir: Path, dpi: int) -> None:
    """Figure S3: friendship probability against home distance."""
    decay = read_aggregate(agg_dir, "distance_decay")
    require_columns(
        decay, ["distance_km", "p_cdr_bay_area", "p_sci_us", "p_sci_bay_area"],
        AGG_FILES["distance_decay"],
    )
    decay = decay.sort_values("distance_km")

    plt.rcParams.update({"font.size": FS})
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    series = [
        ("p_cdr_bay_area", "CDR (Bay Area)", "o-", "tab:blue"),
        ("p_sci_us", "Meta SCI (US)", "s--", "tab:orange"),
        ("p_sci_bay_area", "Meta SCI (Bay Area)", "^-", "tab:green"),
    ]
    for col, label, style, color in series:
        sub = decay.dropna(subset=[col])
        if sub.empty:
            continue
        ax.loglog(sub["distance_km"], sub[col], style, color=color, ms=4, lw=1.8, label=label)

    ax.axhline(1.0, color="gray", linestyle=":", lw=1.2, label="Geographic null")
    ax.set_xlabel("Home distance $d$ (km)", fontsize=FS)
    ax.set_ylabel("$P(d)$ [tie density / geographic null]", fontsize=FS)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS - 2)

    plt.tight_layout()
    out = plots_dir / "figure_s3.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def run_figures_stage(args: argparse.Namespace) -> None:
    """Render the requested figures from the shared aggregates."""
    agg_dir = Path(args.agg_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    renderers = {"2": plot_figure_2, "s3": plot_figure_s3}
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
        help="'figures' (default) draws from the shared aggregates; "
             "'aggregate' rebuilds them from the CDR network and Meta's SCI.",
    )
    parser.add_argument(
        "--figures", nargs="+", choices=["2", "s3"], default=["2", "s3"],
        help="Subset of figures to produce (default: all).",
    )
    parser.add_argument("--data-dir", default=str(here / "data"),
                        help="Repository data directory (default: %(default)s).")
    parser.add_argument("--agg-dir", default=None,
                        help="Directory holding the aggregate tables "
                             "(default: <data-dir>/processed/soccap_comparison).")
    parser.add_argument("--plots-dir", default=str(here / "plots"),
                        help="Directory for the figures (default: %(default)s).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Figure resolution (default: %(default)s).")

    raw = parser.add_argument_group("raw data options (--stage aggregate only)")
    raw.add_argument(
        "--edges-all",
        help="Path to edges_all.parquet, written by "
             "01_processing_cdr.py --stage aggregate.",
    )
    raw.add_argument(
        "--users-zip",
        help="Path to users_with_zip.parquet, written by "
             "01_processing_cdr.py --stage aggregate.",
    )
    raw.add_argument(
        "--social-capital", default=None,
        help="Social Capital Atlas ZIP-level file (default: the copy shipped "
             "in <data-dir>/social_capital).",
    )
    raw.add_argument(
        "--cells-csv",
        help="CSV of cell_id, latitude, longitude, used to locate CDR users "
             "for figure S3.",
    )
    raw.add_argument(
        "--zcta-shapefile",
        help="TIGER/Line ZCTA shapefile, used for ZIP centroids and areas.",
    )
    raw.add_argument(
        "--sci-glob",
        help="Glob matching the Social Connectedness Index shards. Required "
             "for figure S3; figure 2 does not need it.",
    )
    raw.add_argument("--duckdb-threads", type=int, default=8,
                     help="Threads used by DuckDB (default: %(default)s).")

    args = parser.parse_args(argv)
    if args.agg_dir is None:
        args.agg_dir = str(Path(args.data_dir) / "processed" / "soccap_comparison")
    if args.social_capital is None:
        args.social_capital = str(
            Path(args.data_dir) / "social_capital" / "social_capital_zip.csv"
        )

    if args.stage == "aggregate":
        missing = [
            name for name, value in [
                ("--edges-all", args.edges_all), ("--users-zip", args.users_zip),
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
