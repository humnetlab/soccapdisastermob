#!/usr/bin/env python3
"""
07_census_tract_level_associations.py -- the same associations at tract level.

    figure_s10.png  Tract-level connectedness against median household income,
                    before and after residualisation.
    figure_s11.png  Effect of residualised connectedness on evacuation, and the
                    composition of evacuees by income and connectedness.
    figure_s12.png  Association of income and connectedness with whether
                    damaged buildings were abandoned or rebuilt.
    table_s7.csv    Standardised weighted least squares coefficients.

Scripts 05 and 06 measure social capital with the Social Capital Atlas, which is
published at ZIP code level. This script repeats both analyses with a
reconstructed dataset that is native to census tracts, so the findings can be
checked against a finer and independently constructed geography.

Connectedness is again regressed on median household income and the residual
used throughout, since the two are correlated at tract level as well. Recovery
regressions weight tracts by their number of damaged buildings, standardise
within storm, and use HC3 standard errors, as in script 06.

The raw traces cannot be redistributed, so the script runs in two stages:
``--stage figures`` (default) draws from the shared aggregates in ``data/`` and
needs no raw data; ``--stage aggregate`` rebuilds them. Run with --help.
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
from matplotlib.lines import Line2D  # noqa: E402


# --------------------------------------------------------------------------
# Study configuration
# --------------------------------------------------------------------------

#: Storms in the evacuation analysis, in figure order.
HURRICANE_STORMS = [
    "Harvey (2017)", "Irma (2017)", "Florence (2018)",
    "Michael (2018)", "Imelda (2019)",
]

#: Storms with street-view damage assessments, used by the recovery analysis.
DAMAGE_STORMS = ["Harvey", "Irma", "Michael"]

#: Displacement thresholds, in kilometres.
DISTANCE_THRESHOLDS_KM = [10, 50, 100]

#: Quadrants of the income by connectedness split.
GROUP_ORDER = [
    "Low income / Low EC resid",
    "Low income / High EC resid",
    "High income / Low EC resid",
    "High income / High EC resid",
]
GROUP_STYLE = {
    "Low income / Low EC resid": {"color": "#8FD694", "hatch": None},
    "Low income / High EC resid": {"color": "#8FD694", "hatch": "//"},
    "High income / Low EC resid": {"color": "#9EC9FF", "hatch": None},
    "High income / High EC resid": {"color": "#9EC9FF", "hatch": "//"},
}
GROUP_LABEL = {
    "Low income / Low EC resid": "Low Income / Low EC (residual)",
    "Low income / High EC resid": "Low Income / High EC (residual)",
    "High income / Low EC resid": "High Income / Low EC (residual)",
    "High income / High EC resid": "High Income / High EC (residual)",
}

THR_COLORS = {10: "#c44e52", 50: "#8c6bb1", 100: "#2b8cbe"}
THR_OFFSET = {10: 0.22, 50: 0.0, 100: -0.22}

#: Colours of the connectedness against income panels.
RAW_COLOR = "#4C72B0"
RESIDUAL_COLOR = "#F0902B"

#: Recovery outcomes modelled.
OUTCOMES = ["Abandoned_normalized", "R_Higher_normalized"]
OUTCOME_LABEL = {
    "Abandoned_normalized": "% Abandoned",
    "R_Higher_normalized": "% Rebuilt",
}
OUTCOME_COLOR = {
    "Abandoned_normalized": "#1f77b4",
    "R_Higher_normalized": "#2ca02c",
}
OUTCOME_YBASE = {"Abandoned_normalized": 3, "R_Higher_normalized": 1}

#: Social capital metrics, native to census tracts in this dataset. Each is
#: residualised on income before entering the recovery model.
SC_METRICS = ["ec_tract", "volunteering_rate_tract", "support_ratio_tract"]
SC_LABEL = {
    "ec_tract": "Econ. connectedness",
    "volunteering_rate_tract": "Volunteering rate",
    "support_ratio_tract": "Support ratio",
}

PRED_ORDER = ["Median income", "Econ. connectedness (resid)"]
PRED_LABELS = ["Income", "EC (residual)"]

MAIN_THRESHOLD = 480
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 7
MIN_CANDIDATE_H3_MINUTES = 60
HOME_WORK_STAYS = "StayPointsWithHomeWork"

AGG_FILES = {
    "ec_income": "tract_ec_income.csv",
    "ec_effect": "tract_ec_residual_effect.csv",
    "group_share": "tract_group_share_by_threshold.csv",
    "recovery": "tract_recovery_outcomes.csv",
    "income": "tract_median_income.csv",
}

FS = 11


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def log(message: str) -> None:
    print(message, flush=True)


def pad_geoid(values, width: int = 11) -> pd.Series:
    """Normalise census tract GEOIDs to zero-padded digit strings."""
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    return (
        series.astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(width)
    )


def zscore(series: pd.Series) -> pd.Series:
    """Standardise, returning zeros when the series has no variation."""
    sd = float(series.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / sd


def p_to_stars(p: float) -> str:
    """Significance markers used across the figures."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fig_stars(p: float) -> str:
    """Markers used inside the recovery forest plot, matching script 06."""
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares slope and intercept, ignoring non-finite pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        raise ValueError("Need at least two finite observations for a linear fit.")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def assign_income_ec_group(tract_level: pd.DataFrame) -> pd.DataFrame:
    """Split tracts at the median of income and of the EC residual."""
    out = tract_level.copy()
    low_income = out["median_income"] < out["median_income"].median()
    low_ec = out["ec_residual"] < out["ec_residual"].median()
    out["group"] = np.select(
        [low_income & low_ec, low_income & ~low_ec, ~low_income & low_ec],
        [GROUP_ORDER[0], GROUP_ORDER[1], GROUP_ORDER[2]],
        default=GROUP_ORDER[3],
    )
    return out


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
    return pd.read_csv(path, dtype={"GEOID": str})


# --------------------------------------------------------------------------
# Stage 1: aggregation
# --------------------------------------------------------------------------

def build_ec_income_table(
    social_capital_path: Path, income_path: Path,
) -> pd.DataFrame:
    """Residualise tract connectedness against median household income."""
    rec = pd.read_csv(
        social_capital_path, dtype={"census_tract_GEOID": str},
        usecols=["census_tract_GEOID", "EC_main"],
    )
    rec["GEOID"] = pad_geoid(rec["census_tract_GEOID"])
    rec["ec"] = pd.to_numeric(rec["EC_main"], errors="coerce")
    rec = rec[["GEOID", "ec"]].dropna().drop_duplicates("GEOID")

    income = pd.read_csv(income_path, dtype={"GEOID": str})
    income["GEOID"] = pad_geoid(income["GEOID"])

    merged = income.merge(rec, on="GEOID", how="inner")
    fit = merged[["median_income", "ec"]].dropna()
    slope, intercept = linear_fit(
        fit["median_income"].to_numpy(float), fit["ec"].to_numpy(float)
    )
    merged["ec_pred_from_income"] = intercept + slope * merged["median_income"]
    merged["ec_residual"] = merged["ec"] - merged["ec_pred_from_income"]

    log(f"  EC ~ income on {len(fit):,} tracts: "
        f"EC = {intercept:.4f} + {slope:.3e} * income")
    return merged[
        ["GEOID", "median_income", "ec", "ec_pred_from_income", "ec_residual"]
    ]


def build_ec_effect(person: pd.DataFrame, ec_income: pd.DataFrame) -> pd.DataFrame:
    """Regress the tract-level evacuation share on the standardised residual."""
    import statsmodels.api as sm

    rows = []
    for storm in HURRICANE_STORMS:
        sub = person.loc[person["disaster"] == storm].replace(
            [np.inf, -np.inf], np.nan
        )
        sub = sub.merge(
            ec_income[["GEOID", "ec_residual"]], on="GEOID", how="inner"
        ).dropna(subset=["GEOID", "ec_residual"])
        if sub.empty:
            continue

        tract_ec = sub[["GEOID", "ec_residual"]].drop_duplicates("GEOID").dropna()
        tract_ec["ec_resid_z"] = zscore(tract_ec["ec_residual"])

        for threshold in DISTANCE_THRESHOLDS_KM:
            evacuated = (
                (sub["evacuated"] == 1) & (sub["distance_km"] > float(threshold))
            ).astype(float)
            share = (
                sub.assign(evac_ind=evacuated)
                .groupby("GEOID", as_index=False)
                .agg(share=("evac_ind", "mean"))
                .merge(tract_ec, on="GEOID", how="inner")
                .dropna(subset=["share", "ec_resid_z"])
            )
            if len(share) < 3:
                continue

            model = sm.OLS(
                100.0 * share["share"].astype(float),
                sm.add_constant(share["ec_resid_z"].astype(float)),
            ).fit(cov_type="HC1")

            beta = float(model.params.get("ec_resid_z", np.nan))
            se = float(model.bse.get("ec_resid_z", np.nan))
            rows.append({
                "storm": storm,
                "threshold_km": int(threshold),
                "beta_pp": beta,
                "se_pp": se,
                "lo_pp": beta - 1.96 * se,
                "hi_pp": beta + 1.96 * se,
                "p": float(model.pvalues.get("ec_resid_z", np.nan)),
                "n_tract": len(share),
            })
    return pd.DataFrame(rows)


def build_group_shares(person: pd.DataFrame, ec_income: pd.DataFrame) -> pd.DataFrame:
    """Evacuation share of each income by connectedness group, per threshold."""
    rows = []
    for storm in HURRICANE_STORMS:
        sub = person.loc[person["disaster"] == storm].replace(
            [np.inf, -np.inf], np.nan
        )
        sub = sub.merge(
            ec_income[["GEOID", "median_income", "ec_residual"]],
            on="GEOID", how="inner",
        ).dropna(subset=["GEOID", "pid", "median_income", "ec_residual"])
        if sub.empty:
            continue

        tract_level = assign_income_ec_group(
            sub[["GEOID", "median_income", "ec_residual"]]
            .drop_duplicates("GEOID").dropna()
        )
        sub = sub.merge(tract_level[["GEOID", "group"]], on="GEOID", how="inner")

        for threshold in DISTANCE_THRESHOLDS_KM:
            sub[f"y_{threshold}"] = (
                (sub["evacuated"] == 1) & (sub["distance_km"] > float(threshold))
            ).astype(float)

        tract_means = sub.groupby(["GEOID", "group"], as_index=False).agg(
            **{f"share_{t}": (f"y_{t}", "mean") for t in DISTANCE_THRESHOLDS_KM}
        )
        long = tract_means.melt(
            id_vars=["GEOID", "group"],
            value_vars=[f"share_{t}" for t in DISTANCE_THRESHOLDS_KM],
            var_name="metric", value_name="share",
        )
        long["threshold_km"] = long["metric"].str.extract(r"(\d+)").astype(int)

        summary = long.groupby(["group", "threshold_km"], as_index=False).agg(
            mean_share=("share", "mean"),
            sd_share=("share", "std"),
            n_tract=("GEOID", "nunique"),
        )
        summary["se_share"] = (
            summary["sd_share"] / np.sqrt(summary["n_tract"].clip(lower=1))
        )
        summary.insert(0, "storm", storm)
        rows.append(summary)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def map_home_to_tract(
    person: pd.DataFrame, tiger_tracts: Path,
) -> pd.DataFrame:
    """Attach an 11-digit tract GEOID to each user's home ZIP centroid.

    The evacuation cohort is keyed by ZIP code, so homes are located at ZIP
    centroids and joined to the tract containing them.
    """
    import geopandas as gpd

    parts = []
    for path in sorted(tiger_tracts.glob("tl_2020_*_tract")):
        if path.is_dir():
            parts.append(gpd.read_file(path, columns=["GEOID", "geometry"]))
    if not parts:
        raise SystemExit(f"No tl_2020_*_tract directories found in {tiger_tracts}")

    tracts = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), crs=parts[0].crs
    ).to_crs("EPSG:4326")
    tracts = tracts.dropna(subset=["geometry"]).drop_duplicates("GEOID")
    return tracts


def run_aggregate_stage(args: argparse.Namespace) -> None:
    """Rebuild the tract-level aggregates."""
    agg_dir = Path(args.agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)

    log("Residualising tract connectedness against income ...")
    ec_income = build_ec_income_table(
        Path(args.social_capital), agg_dir / AGG_FILES["income"]
    )
    ec_income.to_csv(agg_dir / AGG_FILES["ec_income"], index=False)
    log(f"  wrote {agg_dir / AGG_FILES['ec_income']}")

    person = pd.read_csv(
        args.evacuation_status, dtype={"home_zip": str, "GEOID": str}
    )
    if "GEOID" not in person.columns:
        raise SystemExit(
            "--evacuation-status must carry a GEOID column giving each user's "
            "home census tract. Build it by joining home ZIP centroids to the "
            "2020 tract polygons."
        )
    person["GEOID"] = pad_geoid(person["GEOID"])
    log(f"  evacuation panel: {len(person):,} users")

    log("Estimating the effect of the EC residual on evacuation ...")
    build_ec_effect(person, ec_income).to_csv(
        agg_dir / AGG_FILES["ec_effect"], index=False
    )
    log(f"  wrote {agg_dir / AGG_FILES['ec_effect']}")

    log("Summarising evacuation shares by income and connectedness ...")
    build_group_shares(person, ec_income).to_csv(
        agg_dir / AGG_FILES["group_share"], index=False
    )
    log(f"  wrote {agg_dir / AGG_FILES['group_share']}")


# --------------------------------------------------------------------------
# Recovery regressions
# --------------------------------------------------------------------------

def fit_storm(recovery: pd.DataFrame, outcome: str):
    """Weighted least squares of a recovery outcome on income and connectedness.

    Each social capital metric is residualised on income first. Outcome and
    predictors are standardised, tracts are weighted by their number of damaged
    buildings, and standard errors are HC3.
    """
    import statsmodels.api as sm

    needed = ["median_income", "n_damaged", outcome] + SC_METRICS
    df = recovery.dropna(subset=needed).copy()
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
        result["coef"].map("{:.2f}".format) + result["p"].apply(fig_stars)
    )
    return result, len(df)


def write_table_s7(results: dict, out_dir: Path) -> None:
    """Write the tract-level recovery coefficients, and echo them to stdout."""
    rows = []
    for outcome in OUTCOMES:
        for storm in DAMAGE_STORMS:
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
                    "significance": p_to_stars(row["p"]),
                })

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_s7.csv", index=False)

    log("\nTable S7: tract-level recovery coefficients")
    for outcome in OUTCOMES:
        log(f"  {OUTCOME_LABEL[outcome]}")
        for row in table.loc[table["outcome"] == OUTCOME_LABEL[outcome]].itertuples(
            index=False
        ):
            p = "<0.001" if row.p < 0.001 else f"{row.p:.3f}"
            log(f"    {row.storm:8s} n={row.n:<5d} {row.predictor:28s} "
                f"{row.coef:+.3f}  [{row.ci_low:+.3f}, {row.ci_high:+.3f}]  p={p}")


# --------------------------------------------------------------------------
# Stage 2: figures
# --------------------------------------------------------------------------

def plot_figure_s10(ec_income: pd.DataFrame, plots_dir: Path, dpi: int,
                    restrict: set[str] | None) -> None:
    """Tract connectedness against income, raw and residualised."""
    data = ec_income.dropna(subset=["median_income", "ec", "ec_residual"])
    if restrict:
        subset = data.loc[data["GEOID"].isin(restrict)]
        if len(subset) > 100:
            # Refit within the plotted sample so panel B is orthogonal to income
            # for exactly the tracts drawn.
            data = subset.copy()
            slope, intercept = linear_fit(
                data["median_income"].to_numpy(float), data["ec"].to_numpy(float)
            )
            data["ec_residual"] = (
                data["ec"] - (intercept + slope * data["median_income"])
            )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=dpi)
    income = data["median_income"].to_numpy(float)

    ax = axes[0]
    ax.scatter(income, data["ec"], s=4, alpha=0.35, color=RAW_COLOR,
               edgecolors="none")
    slope, intercept = linear_fit(income, data["ec"].to_numpy(float))
    line_x = np.linspace(income.min(), income.max(), 100)
    ax.plot(line_x, intercept + slope * line_x, color=RAW_COLOR, lw=1.8)
    rho = float(np.corrcoef(income, data["ec"].to_numpy(float))[0, 1])
    ax.set_ylabel("Economic Connectedness", fontsize=12)
    ax.text(0.02, 0.96, "A", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top",
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.5"))
    ax.text(0.97, 0.04, rf"$\rho$ = {rho:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11,
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.5"))

    ax = axes[1]
    residual = data["ec_residual"].to_numpy(float)
    ax.scatter(income, residual, s=4, alpha=0.35, color=RESIDUAL_COLOR,
               edgecolors="none")
    ax.axhline(0.0, color=RESIDUAL_COLOR, ls="--", lw=1.8)
    rho_res = float(np.corrcoef(income, residual)[0, 1])
    ax.set_ylabel("Economic Connectedness (Residual)", fontsize=12)
    ax.text(0.02, 0.96, "B", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top",
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.5"))
    ax.text(0.97, 0.04, rf"$\rho$ = {rho_res:.2e}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11,
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.5"))

    for ax in axes:
        ax.set_xlabel("Median Household Income", fontsize=12)
        ax.set_xticks([80_000, 160_000, 240_000])
        ax.set_xticklabels(["$80,000", "$160,000", "$240,000"])
        ax.grid(True, ls="--", alpha=0.3)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    out = plots_dir / "figure_s10.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def plot_figure_s11(ec_effect: pd.DataFrame, group_share: pd.DataFrame,
                    plots_dir: Path, dpi: int) -> None:
    """Effect of the EC residual on evacuation, and evacuee composition."""
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 5.5), dpi=dpi,
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.25},
    )

    ax_left.axvline(0, color="black", linestyle="--", linewidth=1)
    storms = [s for s in HURRICANE_STORMS if s in set(ec_effect["storm"])]
    if storms:
        span = float(ec_effect["hi_pp"].max() - ec_effect["lo_pp"].min())
        pad = 0.02 * span if np.isfinite(span) and span > 0 else 0.05

        for threshold in DISTANCE_THRESHOLDS_KM:
            sub = (
                ec_effect.loc[ec_effect["threshold_km"] == threshold]
                .set_index("storm").reindex(storms)
            )
            y = np.arange(len(storms)) + THR_OFFSET.get(threshold, 0.0)
            mask = sub["beta_pp"].notna().to_numpy()
            if not mask.any():
                continue
            color = THR_COLORS[threshold]
            ax_left.hlines(y[mask], sub.loc[mask, "lo_pp"], sub.loc[mask, "hi_pp"],
                           color=color, linewidth=2, alpha=0.95)
            ax_left.scatter(sub.loc[mask, "beta_pp"], y[mask], color=color, s=55,
                            label=f">{threshold} km", zorder=3)
            for x_hi, yy, p in zip(sub.loc[mask, "hi_pp"], y[mask], sub.loc[mask, "p"]):
                marker = p_to_stars(p)
                if marker:
                    ax_left.text(float(x_hi) + pad, float(yy), marker, va="center",
                                 ha="left", fontsize=FS, color=color)

        ax_left.set_yticks(np.arange(len(storms)))
        ax_left.set_yticklabels(storms, fontsize=FS)
        ax_left.set_ylabel("Storm", fontsize=FS)
        ax_left.legend(loc="lower right", frameon=True, framealpha=0.92,
                       edgecolor="0.75", fontsize=FS)

    ax_left.set_xlabel("Effect of EC (residual) on Evacuations", fontsize=FS)
    ax_left.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax_left.tick_params(labelsize=FS)
    ax_left.text(-0.12, 1.02, "A", transform=ax_left.transAxes, fontsize=14,
                 fontweight="bold", va="bottom")

    # Right: evacuee composition for the storm with the widest coverage.
    focus = None
    if not group_share.empty:
        counts = group_share.groupby("storm")["n_tract"].sum()
        focus = counts.idxmax()

    if focus is None:
        ax_right.set_visible(False)
    else:
        sub = group_share.loc[group_share["storm"] == focus]
        width = 0.18
        x = np.arange(len(DISTANCE_THRESHOLDS_KM))
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(GROUP_ORDER))

        for offset, group in zip(offsets, GROUP_ORDER):
            series = (
                sub.loc[sub["group"] == group]
                .set_index("threshold_km").reindex(DISTANCE_THRESHOLDS_KM)
            )
            style = GROUP_STYLE.get(group, {"color": "gray", "hatch": None})
            ax_right.bar(x + offset, 100 * series["mean_share"], width=width,
                         color=style["color"], hatch=style["hatch"],
                         edgecolor="black", alpha=0.95,
                         label=GROUP_LABEL.get(group, group))
            ax_right.errorbar(x + offset, 100 * series["mean_share"],
                              yerr=100 * 1.96 * series["se_share"], fmt="none",
                              ecolor="black", elinewidth=1, capsize=3)

        ax_right.set_xticks(x)
        ax_right.set_xticklabels([f">{t} km" for t in DISTANCE_THRESHOLDS_KM],
                                 fontsize=FS)
        ax_right.set_title(focus, fontsize=FS)
        ax_right.set_xlabel("Displacement threshold", fontsize=FS)
        ax_right.set_ylabel("Share of evacuees (%)", fontsize=FS)
        ax_right.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax_right.tick_params(labelsize=FS)
        ax_right.legend(loc="upper right", frameon=True, framealpha=0.92,
                        edgecolor="0.75", fontsize=FS - 2)
        ax_right.text(-0.12, 1.02, "B", transform=ax_right.transAxes, fontsize=14,
                      fontweight="bold", va="bottom")

    out = plots_dir / "figure_s11.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def plot_figure_s12(results: dict, plots_dir: Path, dpi: int) -> None:
    """Recovery coefficients on income and connectedness, per storm."""
    finite = [
        v for outcome in OUTCOMES for storm in DAMAGE_STORMS
        for result, _ in [results[outcome][storm]] if result is not None
        for v in result[["ci_low", "ci_high"]].to_numpy().ravel()
        if np.isfinite(v)
    ]
    xlim = max(0.15, max(abs(v) for v in finite) * 1.40) if finite else 1.0

    fig, axes = plt.subplots(1, len(DAMAGE_STORMS), figsize=(16, 5), dpi=dpi)
    axes = np.atleast_1d(axes)

    for col, storm in enumerate(DAMAGE_STORMS):
        ax = axes[col]
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
                    va="center", fontsize=FS,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.1),
                    clip_on=False,
                )

        ax.set_yticks([3, 2, 1, 0])
        ax.set_yticklabels(PRED_LABELS + PRED_LABELS if col == 0 else [],
                           fontsize=FS + 2)
        ax.set_ylim(-0.6, 3.6)
        ax.set_xlim(-xlim, xlim)
        ax.set_xlabel("Std. coefficient", fontsize=FS + 2)
        ax.set_title(storm, fontsize=FS + 2)
        ax.tick_params(labelsize=FS)

        if col == 0:
            handles = [
                Line2D([0], [0], marker="o", color=OUTCOME_COLOR[o], lw=1.5,
                       markersize=6, label=OUTCOME_LABEL[o])
                for o in OUTCOMES
            ]
            legend = ax.legend(handles=handles, fontsize=FS, frameon=False,
                               loc="upper right")
            for text, outcome in zip(legend.get_texts(), OUTCOMES):
                text.set_color(OUTCOME_COLOR[outcome])

    plt.tight_layout()
    out = plots_dir / "figure_s12.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def run_figures_stage(args: argparse.Namespace) -> None:
    """Produce the three figures and the recovery table."""
    agg_dir = Path(args.agg_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    targets = set(args.figures)

    if "s10" in targets:
        log("Rendering figure S10 ...")
        ec_income = read_aggregate(agg_dir, "ec_income")
        recovery = read_aggregate(agg_dir, "recovery", required=False)
        restrict = (
            set(pad_geoid(recovery["GEOID"])) if not recovery.empty else None
        )
        plot_figure_s10(ec_income, plots_dir, args.dpi, restrict)

    if "s11" in targets:
        log("Rendering figure S11 ...")
        ec_effect = read_aggregate(agg_dir, "ec_effect", required=False)
        group_share = read_aggregate(agg_dir, "group_share", required=False)
        if ec_effect.empty and group_share.empty:
            log("  evacuation aggregates unavailable; figure S11 skipped")
        else:
            plot_figure_s11(ec_effect, group_share, plots_dir, args.dpi)

    if targets & {"s12", "table"}:
        recovery = read_aggregate(agg_dir, "recovery")
        log("Estimating tract-level recovery regressions ...")
        results = {
            outcome: {
                storm: fit_storm(recovery.loc[recovery["event"] == storm], outcome)
                for storm in DAMAGE_STORMS
            }
            for outcome in OUTCOMES
        }
        if "table" in targets:
            write_table_s7(results, plots_dir)
        if "s12" in targets:
            log("\nRendering figure S12 ...")
            plot_figure_s12(results, plots_dir, args.dpi)


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
             "'aggregate' rebuilds them from the raw panel.",
    )
    parser.add_argument(
        "--figures", nargs="+", choices=["s10", "s11", "s12", "table"],
        default=["s10", "s11", "s12", "table"],
        help="Subset of outputs to produce (default: all).",
    )
    parser.add_argument("--data-dir", default=str(here / "data"),
                        help="Repository data directory (default: %(default)s).")
    parser.add_argument("--agg-dir", default=None,
                        help="Directory holding the aggregate tables "
                             "(default: <data-dir>/processed/tract_associations).")
    parser.add_argument("--plots-dir", default=str(here / "plots"),
                        help="Directory for the outputs (default: %(default)s).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Figure resolution (default: %(default)s).")

    raw = parser.add_argument_group("raw data options (--stage aggregate only)")
    raw.add_argument(
        "--evacuation-status",
        help="CSV of per-user evacuation status carrying a GEOID column with "
             "each user's home census tract.",
    )
    raw.add_argument(
        "--social-capital", default=None,
        help="Reconstructed tract-level social capital file (default: the copy "
             "shipped in <data-dir>/social_capital_tract).",
    )
    raw.add_argument(
        "--tiger-tracts",
        help="Directory of tl_2020_XX_tract shapefiles, used to locate homes.",
    )

    args = parser.parse_args(argv)
    if args.agg_dir is None:
        args.agg_dir = str(Path(args.data_dir) / "processed" / "tract_associations")
    if args.social_capital is None:
        args.social_capital = str(
            Path(args.data_dir) / "social_capital_tract"
            / "reconstructed_tract_social_capital.csv"
        )

    if args.stage == "aggregate" and not args.evacuation_status:
        parser.error("--evacuation-status is required when --stage aggregate")

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
