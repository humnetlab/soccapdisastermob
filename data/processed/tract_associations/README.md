# Tract-level associations

Tables behind figures S10-S12 and table S7, produced by
`07_census_tract_level_associations.py`. Every row describes a census tract.

### `tract_ec_income.csv` -- figure S10
Connectedness and income per tract, with the residual used throughout.

| column | description |
|---|---|
| `GEOID` | 11-digit census tract identifier |
| `median_income` | Tract ACS median household income |
| `ec` | Economic connectedness, `EC_main` from the reconstructed dataset |
| `ec_pred_from_income` | Fitted value from regressing `ec` on income |
| `ec_residual` | `ec` minus its fitted value |

### `tract_median_income.csv`
Tract median household income, from the ACS-derived `clh_2022_tract` file.

| column | description |
|---|---|
| `GEOID` | 11-digit census tract identifier |
| `median_income` | Median household income; non-positive values dropped |

### `tract_recovery_outcomes.csv` -- figure S12 and table S7
Tract-level damage and recovery, joined to social capital and income. Buildings
are mapped directly to 2020 tracts, so no ZIP dependency is introduced.

| column | description |
|---|---|
| `event` | `Harvey`, `Irma` or `Michael` |
| `GEOID` | 11-digit census tract identifier |
| `n_total` | Street-view locations observed |
| `n_damaged` | Of those, how many were damaged. Used as the regression weight |
| `n_abandoned`, `n_rhigher` | Damaged buildings abandoned, and rebuilt to an improved structure |
| `Abandoned_normalized` | `n_abandoned` as a share of `n_damaged` |
| `R_Higher_normalized` | `n_rhigher` as a share of `n_damaged` |
| `median_income` | Tract median household income |
| `ec_tract` | Economic connectedness (`EC_main`) |
| `support_ratio_tract` | Support ratio (`SR_main`) |
| `volunteering_rate_tract` | Volunteering rate (`VR_main`) |

### `tract_ec_residual_effect.csv` -- figure S11, left panel
Tract-level evacuation share on the standardised EC residual, one fit per storm
and threshold, HC1 standard errors.

| column | description |
|---|---|
| `storm` | Storm and year |
| `threshold_km` | Minimum displacement for an evacuation to count |
| `beta_pp` | Change in evacuation share, percentage points per SD of the residual |
| `se_pp` | Standard error |
| `lo_pp`, `hi_pp` | 95% confidence interval |
| `p` | Two-sided p-value |
| `n_tract` | Tracts in the regression |

### `tract_group_share_by_threshold.csv` -- figure S11, right panel
Evacuation share of each income by connectedness group. Shares are computed per
tract then averaged, so each tract counts equally.

| column | description |
|---|---|
| `storm` | Storm and year |
| `group` | One of the four income by connectedness quadrants |
| `threshold_km` | Displacement threshold |
| `mean_share` | Mean tract-level evacuation share, as a fraction |
| `sd_share` | Standard deviation across tracts |
| `n_tract` | Tracts in the group |
| `se_share` | Standard error of `mean_share` |
