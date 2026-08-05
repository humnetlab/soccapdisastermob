# Connectedness and evacuation

Tables behind figures S8, 4 and S9, produced by `05_ec_evac_association.py`.
Every row describes a ZIP code or a group of evacuees.

### `zip_ec_income.csv` -- figure S8
Connectedness and income per ZIP code, with the residual used throughout.

| column | description |
|---|---|
| `home_zip` | Five-digit ZIP code, zero-padded |
| `median_income` | ACS 2021 five-year median household income (B19013_001E) |
| `ec` | Economic connectedness, `ec_zip` from the Social Capital Atlas |
| `ec_pred_from_income` | Fitted value from regressing `ec` on income |
| `ec_residual` | `ec` minus its fitted value: connectedness net of income |

Figure S8 is drawn on the ZIP codes entering the analysis, and the residual is
refit within that sample, so panel B's correlation is zero to floating-point
error.

### `ec_residual_effect.csv` -- figure 4, left panel
ZIP-level evacuation share on the standardised EC residual, one fit per storm
and threshold, HC1 standard errors.

| column | description |
|---|---|
| `storm` | Storm and year |
| `threshold_km` | Minimum displacement for an evacuation to count |
| `beta_pp` | Change in evacuation share, percentage points per SD of the residual |
| `se_pp` | Standard error |
| `lo_pp`, `hi_pp` | 95% confidence interval |
| `p` | Two-sided p-value |
| `n_zip` | ZIP codes in the regression |

### `group_share_by_threshold.csv` -- figures 4 and S9, upper panels
Evacuation share of each income by connectedness group. Shares are computed per
ZIP code then averaged, so each ZIP code counts equally.

| column | description |
|---|---|
| `storm` | Storm and year |
| `group` | One of the four income by connectedness quadrants |
| `threshold_km` | Displacement threshold |
| `mean_share` | Mean ZIP-level evacuation share, as a fraction |
| `sd_share` | Standard deviation across ZIP codes |
| `n_zip` | ZIP codes in the group |
| `se_share` | Standard error of `mean_share` |

### `destination_sci_density.csv` -- figures 4 and S9, lower panels
How connected evacuees' destinations were, relative to their origin's average
connectedness. Positive values mean evacuees moved toward places their community
is unusually connected to. Published as kernel densities on a fixed grid rather
than per-evacuee values.

| column | description |
|---|---|
| `storm` | Storm and year |
| `threshold_km` | Displacement threshold |
| `excess_log1p` | Grid point |
| `density` | Density at that point |
| `mean_excess` | Mean excess connectedness for this storm and threshold |
| `n` | Evacuees contributing |
