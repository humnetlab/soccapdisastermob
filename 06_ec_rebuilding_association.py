#!/usr/bin/env python3
"""
06_ec_rebuilding_association.py -- economic connectedness and rebuilding.

    figure_5.png   Cumulative displacement of damaged against undamaged ZIP
                   codes around landfall, and the association of income and
                   connectedness with abandonment and rebuilding.
    table_s5.csv   Standardised weighted least squares coefficients (Eq. 5).
    table_s6.csv   Variance explained by income and by connectedness.

Building outcomes come from street-view imagery captured before and after each
storm, classified as abandoned, rebuilt to an equal structure, or rebuilt to an
improved structure, then aggregated to ZIP codes. Each social capital metric is
residualised on income before entering the model, outcomes and predictors are
standardised within storm, observations are weighted by the number of damaged
buildings, and standard errors are HC3.

The top row tracks the share of users displaced from their pre-disaster home
ZIP code on each night, split by whether that ZIP code had observed damage.
Rates are shown relative to their level ten days before landfall.

Covers Harvey, Irma and Michael, the storms with street-view assessments.

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
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


# --------------------------------------------------------------------------
# Study configuration
# --------------------------------------------------------------------------

#: Storms with street-view damage assessments, in figure order.
DISASTERS = [
    {"label": "Harvey (2017)", "short": "Harvey",
     "dataset": "20170701_20171231_tx_combined", "event_date": date(2017, 8, 25)},
    {"label": "Irma (2017)", "short": "Irma",
     "dataset": "20170701_20171231_fl_combined", "event_date": date(2017, 9, 10)},
    {"label": "Michael (2018)", "short": "Michael",
     "dataset": "20180701_20181231_full_combined", "event_date": date(2018, 10, 10)},
]
STORM_ORDER = [cfg["short"] for cfg in DISASTERS]

#: Days after landfall retained in the daily panel. The figure shows the window
#: from ten days before landfall onward.
DAILY_WIN = 15
PLOT_START_DAY = -10

#: Recovery outcomes modelled, and the colour each is drawn in.
OUTCOMES = ["Abandoned_normalized", "R_Higher_normalized"]
OUTCOME_LABEL = {
    "Abandoned_normalized": "% Abandoned",
    "R_Higher_normalized": "% Rebuilt",
}
OUTCOME_COLOR = {
    "Abandoned_normalized": "#1f77b4",
    "R_Higher_normalized": "#2ca02c",
}
#: Row of the forest plot each outcome's coefficients start on.
OUTCOME_YBASE = {"Abandoned_normalized": 3, "R_Higher_normalized": 1}

#: Social capital metrics residualised on income before entering the model.
SC_METRICS = ["ec_zip", "clustering_zip", "support_ratio_zip"]
SC_LABEL = {
    "ec_zip": "Econ. connectedness",
    "clustering_zip": "Social clustering",
    "support_ratio_zip": "Support ratio",
}

#: Predictors reported in the figure and tables, in order.
PRED_ORDER = ["Median income", "Econ. connectedness (resid)"]
PRED_LABELS = ["Income", "EC (residual)"]

#: Colours of the damaged and undamaged evacuation curves.
GRP_COLORS = {"Damaged": "#de2d26", "Undamaged": "#9e9e9e"}

#: Nighttime window and dwell threshold, matching the evacuation pipeline.
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 7
DAILY_MIN_H3 = 60

HOME_WORK_STAYS = "StayPointsWithHomeWork"

#: Aggregate tables written by ``--stage aggregate``.
AGG_FILES = {
    "zip_recovery": "zip_recovery_outcomes.parquet",
    "zip_coverage": "zip_damage_coverage.csv",
    "evac_timeseries": "evacuation_by_damage_timeseries.csv",
}

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


def zscore(series: pd.Series) -> pd.Series:
    """Standardise to zero mean and unit variance."""
    return (series - series.mean()) / series.std(ddof=0)


def stars(p: float) -> str:
    """Significance markers, as used in the published figure and tables."""
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def table_stars(p: float) -> str:
    """Significance markers for the tables, which use the 0.10 threshold."""
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


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
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"zip": str})


# --------------------------------------------------------------------------
# Regressions
# --------------------------------------------------------------------------

def fit_storm(zip_recovery: pd.DataFrame, outcome: str):
    """Weighted least squares of a recovery outcome on income and connectedness.

    Each social capital metric is residualised on income first, so its
    coefficient is its association net of affluence. Outcome and predictors are
    standardised, observations are weighted by the number of damaged buildings,
    and standard errors are HC3.
    """
    import statsmodels.api as sm

    needed = ["median_income", "n_damaged", outcome] + SC_METRICS
    df = zip_recovery.dropna(subset=needed).copy()
    df = df.loc[df["n_damaged"] >= 1]
    if len(df) < 10:
        return None, len(df)

    for metric in SC_METRICS:
        df[f"{metric}_r"] = sm.OLS(
            df[metric], sm.add_constant(df["median_income"])
        ).fit().resid

    df["y_z"] = zscore(df[outcome])
    df["inc_z"] = zscore(df["median_income"])
    for metric in SC_METRICS:
        df[f"{metric}_rz"] = zscore(df[f"{metric}_r"])

    x_cols = ["inc_z"] + [f"{m}_rz" for m in SC_METRICS]
    labels = {"inc_z": "Median income"}
    labels.update({f"{m}_rz": f"{SC_LABEL[m]} (resid)" for m in SC_METRICS})

    fit = sm.WLS(
        df["y_z"], sm.add_constant(df[x_cols]), weights=df["n_damaged"]
    ).fit(cov_type="HC3")
    conf = fit.conf_int()

    result = pd.DataFrame([
        {
            "predictor": labels[term],
            "coef": fit.params[term],
            "ci_low": conf.loc[term, 0],
            "ci_high": conf.loc[term, 1],
            "p": fit.pvalues[term],
        }
        for term in x_cols
    ])
    result["label_text"] = (
        result["coef"].map("{:.2f}".format) + result["p"].apply(stars)
    )
    return result, len(df)


def variance_decomposition(zip_recovery: pd.DataFrame, outcome: str):
    """Variance in a recovery outcome explained by income and connectedness.

    The income-only and EC-only columns use the raw predictor. The joint model
    uses income plus EC residualised on income, so the increment over the
    income-only model is the unique contribution of connectedness.
    """
    import statsmodels.api as sm

    needed = ["median_income", "n_damaged", outcome, "ec_zip"]
    df = zip_recovery.dropna(subset=needed).copy()
    df = df.loc[df["n_damaged"] >= 1]
    if len(df) < 10:
        return None

    df["y_z"] = zscore(df[outcome])
    df["inc_z"] = zscore(df["median_income"])
    df["ec_z"] = zscore(df["ec_zip"])
    weights = df["n_damaged"]

    def r2(columns, adjusted=False):
        fit = sm.WLS(
            df["y_z"], sm.add_constant(df[columns]), weights=weights
        ).fit()
        return fit.rsquared_adj if adjusted else fit.rsquared

    ec_resid = sm.WLS(
        df["ec_z"], sm.add_constant(df["inc_z"]), weights=weights
    ).fit().resid
    df["ec_rz"] = zscore(ec_resid)

    r2_income = r2(["inc_z"])
    r2_joint = r2(["inc_z", "ec_rz"])
    return {
        "n": len(df),
        "r2_income_only": r2_income,
        "r2_ec_only": r2(["ec_z"]),
        "r2_income_plus_ec": r2_joint,
        "delta_r2_ec": r2_joint - r2_income,
        "adj_r2_joint": r2(["inc_z", "ec_rz"], adjusted=True),
    }


# --------------------------------------------------------------------------
# Stage 1: aggregation from the raw LBS panel
# --------------------------------------------------------------------------

def build_evacuation_timeseries(
    lbs_base_dir: Path, zcta_shapefile: Path, zip_coverage: pd.DataFrame,
    duckdb_threads: int,
) -> pd.DataFrame:
    """Share of users displaced from their home ZIP code, by damage status.

    For each night, a user's dominant nighttime ZIP code is compared with their
    pre-disaster home ZIP code; a user counts as displaced from the first night
    the two differ. Users are grouped by whether their home ZIP code contains
    any observed building damage.
    """
    import duckdb
    import geopandas as gpd
    import h3
    from shapely.geometry import box

    zcta = gpd.read_file(zcta_shapefile)
    zip_col = next(
        (c for c in ["ZCTA5CE20", "ZCTA5CE10", "ZCTA5CE"] if c in zcta.columns),
        None,
    )
    if zip_col is None:
        raise SystemExit(f"No ZCTA identifier column found in {zcta_shapefile}")
    zcta = zcta[[zip_col, "geometry"]].rename(columns={zip_col: "zip"})
    zcta["zip"] = norm_zip(zcta["zip"])
    zcta = zcta.dropna(subset=["geometry"]).drop_duplicates("zip").reset_index(drop=True)
    if zcta.crs is None:
        zcta = zcta.set_crs("EPSG:4326", allow_override=True)

    # Every surveyed ZIP code, damaged or not. The recovery table holds only
    # damaged ZIP codes, so it cannot supply the undamaged comparison group.
    damage_flag = (
        zip_coverage[["event", "zip", "n_damaged"]]
        .assign(damage_grp=lambda d: np.where(d["n_damaged"] > 0, "Damaged", "Undamaged"))
    )
    damage_flag["zip"] = norm_zip(damage_flag["zip"])

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={duckdb_threads}")

    parts = []
    try:
        for cfg in DISASTERS:
            log(f"  {cfg['label']} ...")
            event = pd.Timestamp(cfg["event_date"])
            day_lo = (event + pd.Timedelta(days=PLOT_START_DAY - 30)).date()
            day_hi = (event + pd.Timedelta(days=DAILY_WIN)).date()
            glob_path = str(lbs_base_dir / cfg["dataset"] / HOME_WORK_STAYS / "*.parquet")

            nightly = con.execute(f"""
                WITH night_rows AS (
                    SELECT CAST(caid AS VARCHAR) AS pid,
                           CAST(h3_index AS VARCHAR) AS stay_h3,
                           CASE WHEN hour_of_day < {NIGHT_END_HOUR}
                                THEN CAST(local_time AS DATE) - INTERVAL 1 DAY
                                ELSE CAST(local_time AS DATE) END AS night_date,
                           CAST(stay_duration AS DOUBLE) / 60000000.0 AS night_minutes
                    FROM read_parquet('{glob_path}')
                    WHERE CAST(local_time AS DATE) >= DATE '{day_lo}'
                      AND CAST(local_time AS DATE) <= DATE '{day_hi}'
                      AND stay_duration > 0
                      AND h3_index IS NOT NULL
                      AND (hour_of_day >= {NIGHT_START_HOUR}
                           OR hour_of_day < {NIGHT_END_HOUR})
                )
                SELECT pid, night_date, stay_h3, SUM(night_minutes) AS night_minutes
                FROM night_rows
                GROUP BY 1, 2, 3
            """).df()
            if nightly.empty:
                continue

            nightly = nightly.loc[nightly["night_minutes"] >= DAILY_MIN_H3]
            if nightly.empty:
                continue

            cells = sorted(nightly["stay_h3"].dropna().astype(str).unique().tolist())
            latlon = np.array([h3.cell_to_latlng(c) for c in cells], dtype=float)
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
            lookup = (
                joined[["stay_h3", "zip"]]
                .sort_values(["stay_h3", "zip"], na_position="last")
                .drop_duplicates("stay_h3", keep="first")
            )

            nightly = (
                nightly.merge(lookup, on="stay_h3", how="left")
                .dropna(subset=["zip"])
                .groupby(["pid", "night_date", "zip"], as_index=False)["night_minutes"]
                .sum()
            )
            nightly["night_date"] = pd.to_datetime(nightly["night_date"])

            # The ZIP code where each user spent most of a given night.
            dominant = (
                nightly.sort_values(
                    ["pid", "night_date", "night_minutes", "zip"],
                    ascending=[True, True, False, True],
                )
                .drop_duplicates(["pid", "night_date"], keep="first")
            )

            # Pre-disaster home: the modal dominant ZIP over the four weeks
            # before landfall.
            pre = dominant.loc[
                dominant["night_date"] < event - pd.Timedelta(days=abs(PLOT_START_DAY))
            ]
            home = (
                pre.groupby(["pid", "zip"], as_index=False)["night_minutes"].sum()
                .sort_values(["pid", "night_minutes", "zip"], ascending=[True, False, True])
                .drop_duplicates("pid", keep="first")
                .rename(columns={"zip": "home_zip"})[["pid", "home_zip"]]
            )
            if home.empty:
                continue

            panel = dominant.merge(home, on="pid", how="inner")
            panel["rel_day"] = (panel["night_date"] - event).dt.days
            panel = panel.loc[panel["rel_day"].between(PLOT_START_DAY, DAILY_WIN)]
            panel = panel.merge(
                damage_flag.loc[damage_flag["event"] == cfg["short"],
                                ["zip", "damage_grp"]],
                left_on="home_zip", right_on="zip", how="inner", suffixes=("", "_home"),
            )
            if panel.empty:
                continue

            # Displaced on a night, and displaced at any point up to that night.
            panel["evacuated"] = (panel["zip"] != panel["home_zip"]).astype(int)
            panel = panel.sort_values(["pid", "rel_day"])
            panel["ever_displaced"] = panel.groupby("pid")["evacuated"].cummax()

            series = (
                panel.groupby(["damage_grp", "rel_day"], as_index=False)
                .agg(p=("ever_displaced", "mean"), n=("ever_displaced", "count"))
            )
            series.insert(0, "disaster", cfg["label"])
            parts.append(series)
            log(f"    {panel['pid'].nunique():,} users")
    finally:
        con.close()

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    se = np.sqrt(np.clip(out["p"] * (1 - out["p"]) / np.maximum(out["n"], 1), 0, None))
    out["lo"] = np.clip(out["p"] - 1.96 * se, 0, 1)
    out["hi"] = np.clip(out["p"] + 1.96 * se, 0, 1)
    return out


def run_aggregate_stage(args: argparse.Namespace) -> None:
    """Rebuild the evacuation time series from the proprietary panel."""
    lbs_base = Path(args.lbs_base_dir).expanduser().resolve()
    if not lbs_base.is_dir():
        raise SystemExit(f"--lbs-base-dir does not exist: {lbs_base}")
    if not args.zcta_shapefile:
        raise SystemExit("--zcta-shapefile is required when --stage aggregate")

    agg_dir = Path(args.agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)

    zip_coverage = read_aggregate(agg_dir, "zip_coverage")
    log("Building the evacuation time series by damage status ...")
    series = build_evacuation_timeseries(
        lbs_base, Path(args.zcta_shapefile).expanduser().resolve(),
        zip_coverage, args.duckdb_threads,
    )
    if series.empty:
        raise SystemExit("No evacuation rows were produced.")

    series.to_csv(agg_dir / AGG_FILES["evac_timeseries"], index=False)
    log(f"  wrote {agg_dir / AGG_FILES['evac_timeseries']}")


# --------------------------------------------------------------------------
# Stage 2: tables
# --------------------------------------------------------------------------

def write_table_s5(results: dict, out_dir: Path) -> None:
    """Write table S5 as CSV, and echo it to stdout."""
    rows = []
    for outcome in OUTCOMES:
        for storm in STORM_ORDER:
            result, n = results[outcome][storm]
            if result is None:
                continue
            indexed = result.set_index("predictor")
            for predictor in PRED_ORDER:
                if predictor not in indexed.index:
                    continue
                row = indexed.loc[predictor]
                rows.append({
                    "outcome": OUTCOME_LABEL[outcome],
                    "storm": storm,
                    "n": n,
                    "predictor": predictor,
                    "coef": row["coef"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "p": row["p"],
                })

    table = pd.DataFrame(rows)
    table["significance"] = table["p"].apply(table_stars)
    table.to_csv(out_dir / "table_s5.csv", index=False)

    log("\nTable S5: standardised coefficients by outcome and storm")
    for outcome in OUTCOMES:
        log(f"  {OUTCOME_LABEL[outcome]}")
        sub = table.loc[table["outcome"] == OUTCOME_LABEL[outcome]]
        for row in sub.itertuples(index=False):
            p = "<0.001" if row.p < 0.001 else f"{row.p:.3f}"
            log(f"    {row.storm:8s} n={row.n:<4d} {row.predictor:28s} "
                f"{row.coef:+.3f}  [{row.ci_low:+.3f}, {row.ci_high:+.3f}]  p={p}")


def write_table_s6(zip_recovery: pd.DataFrame, out_dir: Path) -> None:
    """Write table S6 as CSV, and echo it to stdout."""
    rows = []
    for outcome in OUTCOMES:
        for storm in STORM_ORDER:
            stats = variance_decomposition(
                zip_recovery.loc[zip_recovery["event"] == storm], outcome
            )
            if stats is None:
                continue
            rows.append({"outcome": OUTCOME_LABEL[outcome], "storm": storm, **stats})

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_s6.csv", index=False)

    log("\nTable S6: variance explained in recovery outcomes")
    for outcome in OUTCOMES:
        log(f"  {OUTCOME_LABEL[outcome]}")
        for row in table.loc[table["outcome"] == OUTCOME_LABEL[outcome]].itertuples(
            index=False
        ):
            log(f"    {row.storm:8s} n={int(row.n):<4d} income={row.r2_income_only:.3f} "
                f"EC={row.r2_ec_only:.3f} joint={row.r2_income_plus_ec:.3f} "
                f"dR2={row.delta_r2_ec:.3f} adj={row.adj_r2_joint:.3f}")


# --------------------------------------------------------------------------
# Stage 2: figure
# --------------------------------------------------------------------------

def plot_figure_5(
    results: dict, evac_series: pd.DataFrame, plots_dir: Path, dpi: int,
) -> None:
    """Figure 5: displacement by damage status, and the recovery regressions."""
    fig = plt.figure(figsize=(17, 10), dpi=dpi)
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.1, 1.8],
                           hspace=0.50, wspace=0.35)

    # Top row: cumulative displacement, damaged against undamaged ZIP codes.
    for col, cfg in enumerate(DISASTERS):
        ax = fig.add_subplot(gs[0, col])
        sub = evac_series.loc[evac_series["disaster"] == cfg["label"]].sort_values(
            "rel_day"
        )
        for group in ["Undamaged", "Damaged"]:
            series = sub.loc[sub["damage_grp"] == group]
            if series.empty:
                continue
            ax.fill_between(series["rel_day"], series["lo"] * 100, series["hi"] * 100,
                            color=GRP_COLORS[group], alpha=0.18, linewidth=0)
            ax.plot(series["rel_day"], series["p"] * 100, color=GRP_COLORS[group],
                    lw=2, marker="o", ms=3, label=group)

        ax.axvline(0, color="black", ls="--", lw=1, alpha=0.6)
        ax.set_xlim(PLOT_START_DAY, DAILY_WIN)
        ax.set_xlabel("Day relative to landfall", fontsize=FS)
        ax.set_title(cfg["short"], fontsize=FS)
        ax.grid(True, ls="--", alpha=0.4)
        ax.tick_params(labelsize=FS)
        if col == 0:
            ax.set_ylabel("Cumulative\nEvacuation Rate (%)", fontsize=FS)
            ax.legend(fontsize=FS - 4, frameon=False)

    # A common x range keeps the three forest plots comparable.
    finite = [
        v for outcome in OUTCOMES for storm in STORM_ORDER
        for result, _ in [results[outcome][storm]] if result is not None
        for v in result[["ci_low", "ci_high"]].to_numpy().ravel()
        if np.isfinite(v)
    ]
    xlim = max(0.15, max(abs(v) for v in finite) * 1.40) if finite else 1.0

    # Bottom row: coefficients on income and connectedness, per storm.
    for col, storm in enumerate(STORM_ORDER):
        ax = fig.add_subplot(gs[1, col])
        ax.axvline(0, color="#d9d9d9", ls="--", lw=1.0, zorder=0)
        ax.axhline(1.5, color="#cccccc", lw=0.8, ls="-", zorder=0)

        for outcome in OUTCOMES:
            result, _ = results[outcome][storm]
            if result is None:
                continue
            color = OUTCOME_COLOR[outcome]
            ybase = OUTCOME_YBASE[outcome]
            ordered = result.set_index("predictor").reindex(PRED_ORDER).reset_index()

            for i, row in ordered.iterrows():
                if not np.isfinite(row["coef"]):
                    continue
                y = ybase - i
                ax.errorbar(
                    row["coef"], y,
                    xerr=[[row["coef"] - row["ci_low"]], [row["ci_high"] - row["coef"]]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.5,
                    capsize=3, markersize=4.5, zorder=3,
                )
                pad = 0.02 * xlim
                positive = row["coef"] >= 0
                ax.text(
                    row["ci_high"] + pad if positive else row["ci_low"] - pad, y,
                    row["label_text"], ha="left" if positive else "right",
                    va="center", fontsize=FS - 4,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.1),
                    clip_on=False,
                )

        ax.set_yticks([3, 2, 1, 0])
        ax.set_yticklabels(PRED_LABELS + PRED_LABELS if col == 0 else [], fontsize=FS)
        ax.set_ylim(-0.6, 3.6)
        ax.set_xlim(-xlim, xlim)
        ax.set_xlabel("Std. coefficient", fontsize=FS)
        ax.set_title(storm, fontsize=FS)
        ax.tick_params(labelsize=FS)

        if col == 0:
            handles = [
                Line2D([0], [0], marker="o", color=OUTCOME_COLOR[o], lw=1.5,
                       markersize=6, label=OUTCOME_LABEL[o])
                for o in OUTCOMES
            ]
            legend = ax.legend(handles=handles, fontsize=FS - 4, frameon=False,
                               loc="upper right")
            for text, outcome in zip(legend.get_texts(), OUTCOMES):
                text.set_color(OUTCOME_COLOR[outcome])

    plt.tight_layout()
    out = plots_dir / "figure_5.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def rebase_to_start(series: pd.DataFrame) -> pd.DataFrame:
    """Express displacement relative to its level at the start of the window.

    Each group's value on the first plotted day is subtracted from the curve and
    its interval, so the panels show displacement accumulated over the event
    rather than the pre-existing baseline.
    """
    if series.empty:
        return series
    out = series.copy()
    base = (
        out.loc[out["rel_day"] == PLOT_START_DAY, ["disaster", "damage_grp", "p"]]
        .rename(columns={"p": "p_base"})
    )
    out = out.merge(base, on=["disaster", "damage_grp"], how="left")
    for column in ["p", "lo", "hi"]:
        out[column] = (out[column] - out["p_base"]).clip(lower=0)
    return out.drop(columns=["p_base"])


def run_figures_stage(args: argparse.Namespace) -> None:
    """Produce the figure and both tables from the shared aggregates."""
    agg_dir = Path(args.agg_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    zip_recovery = read_aggregate(agg_dir, "zip_recovery")
    for column in ["event", "zip", "n_damaged", "median_income"] + SC_METRICS:
        if column not in zip_recovery.columns:
            raise SystemExit(
                f"{AGG_FILES['zip_recovery']} is missing the '{column}' column; "
                "found: " + ", ".join(zip_recovery.columns)
            )

    log("Estimating recovery regressions ...")
    results = {
        outcome: {
            storm: fit_storm(zip_recovery.loc[zip_recovery["event"] == storm], outcome)
            for storm in STORM_ORDER
        }
        for outcome in OUTCOMES
    }

    if not args.skip_tables:
        write_table_s5(results, plots_dir)
        write_table_s6(zip_recovery, plots_dir)

    if not args.skip_figure:
        log("\nRendering figure 5 ...")
        evac_series = read_aggregate(agg_dir, "evac_timeseries", required=False)
        if evac_series.empty:
            log("  evacuation time series unavailable; top row will be blank")
            evac_series = pd.DataFrame(
                columns=["disaster", "damage_grp", "rel_day", "p", "lo", "hi", "n"]
            )
        else:
            evac_series = rebase_to_start(evac_series)
        plot_figure_5(results, evac_series, plots_dir, args.dpi)


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
             "shared aggregates; 'aggregate' rebuilds the evacuation panel.",
    )
    parser.add_argument("--data-dir", default=str(here / "data"),
                        help="Repository data directory (default: %(default)s).")
    parser.add_argument("--agg-dir", default=None,
                        help="Directory holding the aggregate tables "
                             "(default: <data-dir>/rebuilding).")
    parser.add_argument("--plots-dir", default=str(here / "plots"),
                        help="Directory for the figure and tables "
                             "(default: %(default)s).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Figure resolution (default: %(default)s).")
    parser.add_argument("--skip-figure", action="store_true",
                        help="Produce only the regression tables.")
    parser.add_argument("--skip-tables", action="store_true",
                        help="Produce only the figure.")

    raw = parser.add_argument_group("raw data options (--stage aggregate only)")
    raw.add_argument(
        "--lbs-base-dir",
        help="Directory containing the processed stay-point datasets. The raw "
             "LBS data is proprietary and is not distributed with this "
             "repository.",
    )
    raw.add_argument("--zcta-shapefile",
                     help="TIGER/Line ZCTA shapefile.")
    raw.add_argument("--duckdb-threads", type=int, default=8,
                     help="Threads used by DuckDB (default: %(default)s).")

    args = parser.parse_args(argv)
    if args.agg_dir is None:
        args.agg_dir = str(Path(args.data_dir) / "rebuilding")

    if args.stage == "aggregate" and not args.lbs_base_dir:
        parser.error("--lbs-base-dir is required when --stage aggregate")

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
