#!/usr/bin/env python3
"""
05_ec_evac_association.py -- economic connectedness and evacuation.

    figure_s8.png  Economic connectedness against median household income,
                   before and after residualisation.
    figure_4.png   Effect of residualised connectedness on evacuation, the
                   composition of evacuees by income and connectedness, and the
                   social connectedness of the places evacuees went.
    figure_s9.png  The latter two panels, for every storm.

Connectedness is strongly correlated with income, so it is regressed on median
household income across ZIP codes and the residual used throughout; figure S8
confirms the residual is orthogonal to income. Evacuation is measured as in
``03_processing_lbs.py``, split further at 10, 50 and 100 km of displacement.

The left panel of figure 4 regresses the ZIP-level evacuation share on the
standardised residual with HC1 standard errors. The upper right panel splits
ZIP codes at the median of income and of the residual. The lower right panel
compares the Social Connectedness Index between each evacuee's origin and
destination against that origin's average connectedness, so positive values
mean evacuees moved toward places their community is unusually connected to.

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


# --------------------------------------------------------------------------
# Study configuration
# --------------------------------------------------------------------------

#: Storms, in the order they appear across the figures.
HURRICANE_STORMS = [
    "Harvey (2017)", "Irma (2017)", "Florence (2018)",
    "Michael (2018)", "Imelda (2019)",
]

#: Displacement thresholds, in kilometres, defining "evacuated at least this far".
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

#: Colour per displacement threshold, shared by the coefficient and density panels.
THR_COLORS = {10: "#c44e52", 50: "#8c6bb1", 100: "#2b8cbe"}

#: Vertical offset of each threshold within a storm's row of the forest plot.
THR_OFFSET = {10: 0.22, 50: 0.0, 100: -0.22}

#: Colours of figure S8, matching the published panels.
S8_RAW_COLOR = "#4C72B0"
S8_RESIDUAL_COLOR = "#F0902B"

#: Storm used for the two right-hand panels of figure 4.
FOCUS_STORM = "Florence (2018)"

FS = 9

#: Aggregate tables written by ``--stage aggregate``.
AGG_FILES = {
    "ec_income": "zip_ec_income.csv",
    "ec_effect": "ec_residual_effect.csv",
    "group_share": "group_share_by_threshold.csv",
    "sci_density": "destination_sci_density.csv",
}


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
    """Standardise, returning zeros when the series has no variation."""
    sd = float(series.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / sd


def p_to_stars(p: float) -> str:
    """Significance markers, as used in the published figure."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares slope and intercept, ignoring non-finite pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        raise ValueError("Need at least two finite observations for a linear fit.")
    if np.nanstd(x) == 0:
        raise ValueError("No variation in x; cannot fit a regression.")
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    return slope, y.mean() - slope * x.mean()


def assign_income_ec_group(zip_level: pd.DataFrame) -> pd.DataFrame:
    """Split ZIP codes at the median of income and of the EC residual."""
    out = zip_level.copy()
    income_median = out["median_income"].median()
    ec_median = out["ec_residual"].median()

    low_income = out["median_income"] < income_median
    low_ec = out["ec_residual"] < ec_median

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
    return pd.read_csv(path, dtype={"home_zip": str})


# --------------------------------------------------------------------------
# Stage 1: aggregation
# --------------------------------------------------------------------------

def build_ec_income_table(
    social_capital_path: Path, income_lookup: Path,
) -> pd.DataFrame:
    """Residualise economic connectedness against median household income.

    EC is strongly correlated with income, so the residual from a linear fit is
    used throughout as a measure of connectedness net of affluence.
    """
    ec = pd.read_csv(social_capital_path, dtype={"zip": str})
    ec["home_zip"] = norm_zip(ec["zip"])
    ec["ec"] = pd.to_numeric(ec["ec_zip"], errors="coerce")
    ec = ec[["home_zip", "ec"]].dropna(subset=["ec"])
    ec = ec.groupby("home_zip", as_index=False)["ec"].mean()

    income = pd.read_csv(income_lookup, dtype={"home_zip": str})
    income["home_zip"] = norm_zip(income["home_zip"])

    merged = income.merge(ec, on="home_zip", how="inner")
    fit = merged[["median_income", "ec"]].dropna()
    slope, intercept = linear_fit(
        fit["median_income"].to_numpy(float), fit["ec"].to_numpy(float)
    )
    merged["ec_pred_from_income"] = intercept + slope * merged["median_income"]
    merged["ec_residual"] = merged["ec"] - merged["ec_pred_from_income"]

    log(f"  EC ~ income on {len(fit):,} ZIP codes: "
        f"EC = {intercept:.4f} + {slope:.3e} * income")
    return merged[
        ["home_zip", "median_income", "ec", "ec_pred_from_income", "ec_residual"]
    ]


def build_ec_effect(person: pd.DataFrame, ec_income: pd.DataFrame) -> pd.DataFrame:
    """Regress the ZIP-level evacuation share on the standardised EC residual.

    One regression per storm and displacement threshold, on ZIP codes rather
    than users, with heteroskedasticity-robust (HC1) standard errors. The
    coefficient is the change in the evacuation share, in percentage points,
    per standard deviation of the EC residual.
    """
    import statsmodels.api as sm

    rows = []
    for storm in HURRICANE_STORMS:
        sub = person.loc[person["disaster"] == storm].replace(
            [np.inf, -np.inf], np.nan
        )
        sub = sub.merge(ec_income[["home_zip", "ec_residual"]], on="home_zip",
                        how="inner").dropna(subset=["home_zip", "ec_residual"])
        if sub.empty:
            continue

        zip_ec = sub[["home_zip", "ec_residual"]].drop_duplicates("home_zip").dropna()
        zip_ec["ec_resid_z"] = zscore(zip_ec["ec_residual"])

        for threshold in DISTANCE_THRESHOLDS_KM:
            evacuated = (
                (sub["evacuated"] == 1) & (sub["distance_km"] > float(threshold))
            ).astype(float)
            share = (
                sub.assign(evac_ind=evacuated)
                .groupby("home_zip", as_index=False)
                .agg(share=("evac_ind", "mean"))
                .merge(zip_ec, on="home_zip", how="inner")
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
                "n_zip": len(share),
            })
    return pd.DataFrame(rows)


def build_group_shares(person: pd.DataFrame, ec_income: pd.DataFrame) -> pd.DataFrame:
    """Evacuation share of each income by connectedness group, per threshold.

    Shares are computed per ZIP code and then averaged within a group, so that
    every ZIP code contributes equally regardless of how many users it holds.
    """
    rows = []
    for storm in HURRICANE_STORMS:
        sub = person.loc[person["disaster"] == storm].replace(
            [np.inf, -np.inf], np.nan
        )
        sub = sub.merge(
            ec_income[["home_zip", "median_income", "ec_residual"]],
            on="home_zip", how="inner",
        ).dropna(subset=["home_zip", "pid", "median_income", "ec_residual"])
        if sub.empty:
            continue

        zip_level = assign_income_ec_group(
            sub[["home_zip", "median_income", "ec_residual"]]
            .drop_duplicates("home_zip").dropna()
        )
        sub = sub.merge(zip_level[["home_zip", "group"]], on="home_zip", how="inner")

        for threshold in DISTANCE_THRESHOLDS_KM:
            sub[f"y_{threshold}"] = (
                (sub["evacuated"] == 1) & (sub["distance_km"] > float(threshold))
            ).astype(float)

        zip_means = sub.groupby(["home_zip", "group"], as_index=False).agg(
            **{f"share_{t}": (f"y_{t}", "mean") for t in DISTANCE_THRESHOLDS_KM}
        )
        long = zip_means.melt(
            id_vars=["home_zip", "group"],
            value_vars=[f"share_{t}" for t in DISTANCE_THRESHOLDS_KM],
            var_name="metric", value_name="share",
        )
        long["threshold_km"] = long["metric"].str.extract(r"(\d+)").astype(int)

        summary = long.groupby(["group", "threshold_km"], as_index=False).agg(
            mean_share=("share", "mean"),
            sd_share=("share", "std"),
            n_zip=("home_zip", "nunique"),
        )
        summary["se_share"] = (
            summary["sd_share"] / np.sqrt(summary["n_zip"].clip(lower=1))
        )
        summary.insert(0, "storm", storm)
        rows.append(summary)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_sci_density(
    person: pd.DataFrame, sci_glob: str, duckdb_threads: int,
) -> pd.DataFrame:
    """Density of the destination's excess social connectedness, per storm.

    For each evacuee, the Social Connectedness Index between their origin and
    destination is compared with the mean connectedness of that origin to all
    other ZIP codes. The published curves are kernel densities, so densities
    are evaluated here on a fixed grid and only those are written out; no
    per-evacuee value is released.
    """
    import duckdb
    from scipy.stats import gaussian_kde

    evacuees = person.loc[
        (person["evacuated"] == 1)
        & person["home_zip"].notna()
        & person["dest_zip"].notna()
        & person["distance_km"].notna()
    ].copy()
    if evacuees.empty:
        return pd.DataFrame()

    origins = sorted(evacuees["home_zip"].dropna().unique().tolist())
    log(f"  querying SCI for {len(origins):,} origin ZIP codes ...")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={duckdb_threads}")
    con.register("origin_zips", pd.DataFrame({"home_zip": origins}))
    sci = con.execute(f"""
        SELECT CAST(user_region AS VARCHAR) AS home_zip,
               CAST(friend_region AS VARCHAR) AS dest_zip,
               scaled_sci
        FROM read_csv_auto('{sci_glob}', header=true)
        WHERE user_country = 'US' AND friend_country = 'US'
          AND CAST(user_region AS VARCHAR) IN (SELECT home_zip FROM origin_zips)
    """).df()
    con.unregister("origin_zips")
    con.close()

    sci["home_zip"] = norm_zip(sci["home_zip"])
    sci["dest_zip"] = norm_zip(sci["dest_zip"])

    # Each origin's average connectedness, the baseline a destination is
    # judged against.
    baseline = sci.groupby("home_zip")["scaled_sci"].mean().rename("sci_baseline")

    rows = []
    # Wide enough to contain the tails at every threshold; the figure trims to
    # where each storm's densities actually have mass.
    grid = np.linspace(-6.0, 8.0, 500)
    for storm in HURRICANE_STORMS:
        storm_evac = evacuees.loc[evacuees["disaster"] == storm]
        if storm_evac.empty:
            continue
        for threshold in DISTANCE_THRESHOLDS_KM:
            far = storm_evac.loc[storm_evac["distance_km"] > float(threshold)]
            if far.empty:
                continue

            joined = (
                far.merge(sci, on=["home_zip", "dest_zip"], how="left")
                .merge(baseline, on="home_zip", how="left")
                .dropna(subset=["scaled_sci", "sci_baseline"])
            )
            if len(joined) < 2:
                continue

            excess = (
                np.log1p(joined["scaled_sci"].clip(lower=0))
                - np.log1p(joined["sci_baseline"].clip(lower=0))
            ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(excess) < 2 or np.unique(excess).size < 2:
                continue

            density = gaussian_kde(excess)(grid)
            rows.append(pd.DataFrame({
                "storm": storm,
                "threshold_km": int(threshold),
                "excess_log1p": grid,
                "density": density,
                "mean_excess": float(np.mean(excess)),
                "n": len(excess),
            }))
            log(f"    {storm}, >{threshold} km: {len(excess):,} evacuees, "
                f"mean {np.mean(excess):+.3f}")

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_aggregate_stage(args: argparse.Namespace) -> None:
    """Rebuild every aggregate table from the raw inputs."""
    agg_dir = Path(args.agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)

    log("Residualising economic connectedness against income ...")
    ec_income = build_ec_income_table(
        Path(args.social_capital), Path(args.income_lookup)
    )
    ec_income.to_csv(agg_dir / AGG_FILES["ec_income"], index=False)
    log(f"  wrote {agg_dir / AGG_FILES['ec_income']}")

    person = pd.read_csv(
        args.evacuation_status, dtype={"home_zip": str, "dest_zip": str}
    )
    person["home_zip"] = norm_zip(person["home_zip"])
    person.loc[person["dest_zip"].notna(), "dest_zip"] = norm_zip(
        person.loc[person["dest_zip"].notna(), "dest_zip"]
    )
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

    if args.sci_glob:
        log("Computing destination connectedness densities ...")
        density = build_sci_density(person, args.sci_glob, args.duckdb_threads)
        if not density.empty:
            density.to_csv(agg_dir / AGG_FILES["sci_density"], index=False)
            log(f"  wrote {agg_dir / AGG_FILES['sci_density']}")
    else:
        log("  --sci-glob not given; skipping destination connectedness")


# --------------------------------------------------------------------------
# Stage 2: figures
# --------------------------------------------------------------------------

def plot_figure_s8(ec_income: pd.DataFrame, plots_dir: Path, dpi: int,
                   restrict_zips: set[str] | None) -> None:
    """Figure S8: EC against income, raw and residualised."""
    data = ec_income.dropna(subset=["median_income", "ec", "ec_residual"])
    if restrict_zips:
        subset = data.loc[data["home_zip"].isin(restrict_zips)]
        if len(subset) > 100:
            # Refit within the plotted sample, so panel B shows a residual that
            # is orthogonal to income for exactly the ZIP codes drawn.
            data = subset.copy()
            slope, intercept = linear_fit(
                data["median_income"].to_numpy(float), data["ec"].to_numpy(float)
            )
            data["ec_residual"] = (
                data["ec"] - (intercept + slope * data["median_income"])
            )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=dpi)

    income = data["median_income"].to_numpy(float)
    ticks = [80_000, 160_000, 240_000]
    tick_labels = ["$80,000", "$160,000", "$240,000"]

    # Left: the raw relationship, with the fit that defines the residual.
    ax = axes[0]
    ax.scatter(income, data["ec"], s=6, alpha=0.45, color=S8_RAW_COLOR,
               edgecolors="none")
    slope, intercept = linear_fit(income, data["ec"].to_numpy(float))
    line_x = np.linspace(income.min(), income.max(), 100)
    ax.plot(line_x, intercept + slope * line_x, color=S8_RAW_COLOR, lw=1.8)
    rho_raw = float(np.corrcoef(income, data["ec"].to_numpy(float))[0, 1])
    ax.set_ylabel("Economic Connectedness", fontsize=12)
    ax.text(0.02, 0.96, "A", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top",
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.5"))
    ax.text(0.97, 0.04, rf"$\rho$ = {rho_raw:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11,
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.5"))

    # Right: the residual, which is orthogonal to income by construction.
    ax = axes[1]
    residual = data["ec_residual"].to_numpy(float)
    ax.scatter(income, residual, s=6, alpha=0.45, color=S8_RESIDUAL_COLOR,
               edgecolors="none")
    ax.axhline(0.0, color=S8_RESIDUAL_COLOR, ls="--", lw=1.8)
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
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, ls="--", alpha=0.3)
        ax.tick_params(labelsize=10)

    plt.tight_layout()
    out = plots_dir / "figure_s8.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def draw_effect_panel(ax, ec_effect: pd.DataFrame) -> None:
    """Forest plot of the EC residual effect, by storm and threshold."""
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    if ec_effect.empty:
        ax.set_visible(False)
        return

    storms = [s for s in HURRICANE_STORMS if s in set(ec_effect["storm"])]
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
        ax.hlines(y[mask], sub.loc[mask, "lo_pp"], sub.loc[mask, "hi_pp"],
                  color=color, linewidth=2, alpha=0.95)
        ax.scatter(sub.loc[mask, "beta_pp"], y[mask], color=color, s=55,
                   label=f">{threshold} km", zorder=3)

        for x_hi, yy, p in zip(sub.loc[mask, "hi_pp"], y[mask], sub.loc[mask, "p"]):
            stars = p_to_stars(p)
            if stars:
                ax.text(float(x_hi) + pad, float(yy), stars, va="center",
                        ha="left", fontsize=FS, color=color)

    ax.set_yticks(np.arange(len(storms)))
    ax.set_yticklabels(storms, fontsize=FS)
    ax.set_ylabel("Storm", fontsize=FS)
    ax.set_xlabel("Effect of EC (residual) on Evacuations", fontsize=FS)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=FS)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.50, 0.80),
                  bbox_transform=ax.transAxes, ncol=1, frameon=True,
                  framealpha=0.92, edgecolor="0.75", fontsize=FS)
        # Placed low-left, where no storm's interval reaches.
        ax.text(0.02, 0.02, "* p<0.10\n** p<0.05\n*** p<0.01",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=FS,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="0.75", alpha=0.92))


def draw_group_panel(ax, group_share: pd.DataFrame, storm: str,
                     show_legend: bool, show_ylabel: bool,
                     with_errorbars: bool = False) -> None:
    """Evacuation share of each income by connectedness group."""
    sub = group_share.loc[group_share["storm"] == storm]
    if sub.empty:
        ax.set_visible(False)
        return

    width = 0.18
    x = np.arange(len(DISTANCE_THRESHOLDS_KM))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(GROUP_ORDER))

    for offset, group in zip(offsets, GROUP_ORDER):
        series = (
            sub.loc[sub["group"] == group]
            .set_index("threshold_km").reindex(DISTANCE_THRESHOLDS_KM)
        )
        style = GROUP_STYLE.get(group, {"color": "gray", "hatch": None})
        ax.bar(x + offset, 100 * series["mean_share"], width=width,
               color=style["color"], hatch=style["hatch"], edgecolor="black",
               alpha=0.95, label=GROUP_LABEL.get(group, group))
        if with_errorbars and "se_share" in series:
            ax.errorbar(x + offset, 100 * series["mean_share"],
                        yerr=100 * 1.96 * series["se_share"], fmt="none",
                        ecolor="black", elinewidth=1, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f">{t} km" for t in DISTANCE_THRESHOLDS_KM], fontsize=FS)
    ax.set_title(storm, fontsize=FS)
    ax.set_xlabel("Displacement threshold", fontsize=FS)
    if show_ylabel:
        ax.set_ylabel("Share of evacuees (%)", fontsize=FS)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(labelsize=FS)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.72, 1.0), ncol=1, frameon=True,
                      framealpha=0.92, edgecolor="0.75", fontsize=FS - 2)


def draw_sci_panel(ax, sci_density: pd.DataFrame, storm: str,
                   show_ylabel: bool) -> None:
    """Density of the destination's excess social connectedness."""
    sub = sci_density.loc[sci_density["storm"] == storm]
    if sub.empty:
        ax.set_visible(False)
        return

    for threshold in DISTANCE_THRESHOLDS_KM:
        series = sub.loc[sub["threshold_km"] == threshold].sort_values("excess_log1p")
        if series.empty:
            continue
        color = THR_COLORS[threshold]
        ax.fill_between(series["excess_log1p"], series["density"], color=color,
                        alpha=0.38, lw=1.2, label=f">{threshold} km")
        ax.plot(series["excess_log1p"], series["density"], color=color, lw=1.2)
        ax.axvline(float(series["mean_excess"].iloc[0]), color=color, ls="--",
                   lw=1.0, alpha=0.95)

    ax.axvline(0.0, color="black", lw=0.9)

    # Trim the axis to where the densities actually have mass.
    with_mass = sub.loc[sub["density"] > sub["density"].max() * 0.005, "excess_log1p"]
    if not with_mass.empty:
        lo, hi = float(with_mass.min()), float(with_mass.max())
        pad = 0.08 * max(hi - lo, 0.25)
        ax.set_xlim(lo - pad, hi + pad)

    ax.set_xlabel(
        r"$\log\!\left(\frac{\mathrm{SCI}_{dest}}{\mathrm{SCI}_{mean}}\right)$",
        fontsize=FS,
    )
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=FS)
    ax.set_title(storm, fontsize=FS)
    ax.tick_params(labelsize=FS)
    ax.legend(frameon=False, loc="upper right", fontsize=FS)


def plot_figure_4(ec_effect: pd.DataFrame, group_share: pd.DataFrame,
                  sci_density: pd.DataFrame, plots_dir: Path, dpi: int) -> None:
    """Figure 4: the EC effect, evacuee composition and destination choice."""
    fig = plt.figure(figsize=(11.5, 6.2), dpi=dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0],
                          height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.42)

    draw_effect_panel(fig.add_subplot(gs[:, 0]), ec_effect)
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_br = fig.add_subplot(gs[1, 1])

    draw_group_panel(ax_tr, group_share, FOCUS_STORM,
                     show_legend=True, show_ylabel=True)
    if sci_density.empty:
        log("  destination connectedness unavailable; lower right panel omitted")
        ax_br.set_visible(False)
    else:
        draw_sci_panel(ax_br, sci_density, FOCUS_STORM, show_ylabel=True)

    for ax in (ax_tr, ax_br):
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)

    out = plots_dir / "figure_4.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def plot_figure_s9(group_share: pd.DataFrame, sci_density: pd.DataFrame,
                   plots_dir: Path, dpi: int) -> None:
    """Figure S9: the right-hand panels of figure 4, for every storm."""
    storms = [s for s in HURRICANE_STORMS if s in set(group_share["storm"])]
    if not storms:
        log("  no storms available; figure S9 skipped")
        return

    fig = plt.figure(figsize=(4.2 * len(storms), 6.2), dpi=dpi)
    gs = fig.add_gridspec(2, len(storms), height_ratios=[1.0, 1.0],
                          hspace=0.42, wspace=0.25)

    for j, storm in enumerate(storms):
        ax_tr = fig.add_subplot(gs[0, j])
        ax_br = fig.add_subplot(gs[1, j])

        draw_group_panel(ax_tr, group_share, storm, show_legend=(j == 0),
                         show_ylabel=(j == 0), with_errorbars=True)
        if sci_density.empty:
            ax_br.set_visible(False)
        else:
            draw_sci_panel(ax_br, sci_density, storm, show_ylabel=(j == 0))

        for ax in (ax_tr, ax_br):
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(True)

    plt.tight_layout()
    out = plots_dir / "figure_s9.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out}")


def run_figures_stage(args: argparse.Namespace) -> None:
    """Produce all three figures from the shared aggregates."""
    agg_dir = Path(args.agg_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    targets = set(args.figures)

    if "s8" in targets:
        log("Rendering figure S8 ...")
        ec_income = read_aggregate(agg_dir, "ec_income")
        # The published panel is drawn on the ZIP codes entering the analysis,
        # not on every ZIP code in the country.
        restrict = None
        intensity_path = Path(args.data_dir) / "rg_variation" / "zip_intensity.csv"
        if intensity_path.exists():
            intensity = pd.read_csv(intensity_path, dtype={"home_zip": str})
            restrict = set(norm_zip(intensity["home_zip"]))
        plot_figure_s8(ec_income, plots_dir, args.dpi, restrict)

    if targets & {"4", "s9"}:
        ec_effect = read_aggregate(agg_dir, "ec_effect")
        group_share = read_aggregate(agg_dir, "group_share")
        sci_density = read_aggregate(agg_dir, "sci_density", required=False)

        if "4" in targets:
            log("Rendering figure 4 ...")
            plot_figure_4(ec_effect, group_share, sci_density, plots_dir, args.dpi)
        if "s9" in targets:
            log("Rendering figure S9 ...")
            plot_figure_s9(group_share, sci_density, plots_dir, args.dpi)


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
        help="'figures' (default) draws the figures from the shared "
             "aggregates; 'aggregate' rebuilds those aggregates.",
    )
    parser.add_argument(
        "--figures", nargs="+", choices=["s8", "4", "s9"],
        default=["s8", "4", "s9"],
        help="Subset of figures to produce (default: all).",
    )
    parser.add_argument("--data-dir", default=str(here / "data"),
                        help="Repository data directory (default: %(default)s).")
    parser.add_argument("--agg-dir", default=None,
                        help="Directory holding the aggregate tables "
                             "(default: <data-dir>/ec_evac).")
    parser.add_argument("--plots-dir", default=str(here / "plots"),
                        help="Directory for the figures (default: %(default)s).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Figure resolution (default: %(default)s).")

    raw = parser.add_argument_group("raw data options (--stage aggregate only)")
    raw.add_argument(
        "--evacuation-status",
        help="CSV of per-user evacuation status with home_zip, dest_zip, "
             "evacuated and distance_km, as written by "
             "03_processing_lbs.py --write-evacuation-cohort.",
    )
    raw.add_argument(
        "--social-capital", default=None,
        help="Social Capital Atlas ZIP-level file (default: the copy shipped "
             "in <data-dir>/social_capital).",
    )
    raw.add_argument(
        "--income-lookup",
        help="CSV of home_zip and median_income (ACS table B19013_001E).",
    )
    raw.add_argument(
        "--sci-glob",
        help="Glob matching the Social Connectedness Index shards. Required "
             "for the destination connectedness panels.",
    )
    raw.add_argument("--duckdb-threads", type=int, default=8,
                     help="Threads used by DuckDB (default: %(default)s).")

    args = parser.parse_args(argv)
    if args.agg_dir is None:
        args.agg_dir = str(Path(args.data_dir) / "ec_evac")
    if args.social_capital is None:
        args.social_capital = str(
            Path(args.data_dir) / "social_capital" / "social_capital_zip.csv"
        )

    if args.stage == "aggregate":
        missing = [
            name for name, value in [
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
