# Rebuilding outcomes

Tables behind figure 5 and tables S5/S6, produced by
`06_ec_rebuilding_association.py`. Covers Harvey, Irma and Michael, the storms
with street-view damage assessments.

### `zip_recovery_outcomes.parquet` -- tables S5 and S6
ZIP-level damage and recovery joined to social capital and income. Contains only
ZIP codes with at least one damaged building, which are the regression sample.

Outcomes come from street-view imagery captured before and after each storm. The
`_normalized` columns express each category as a share of damaged buildings; the
`_overall` columns use all observed buildings and are kept for reference.

| column | description |
|---|---|
| `event` | `Harvey`, `Irma` or `Michael` |
| `zip` | Five-digit ZIP code |
| `n_total` | Street-view locations observed |
| `n_damaged` | Of those, how many were damaged. Used as the regression weight |
| `damage_share_overall` | `n_damaged` as a share of `n_total` |
| `Abandoned_normalized` | Share of damaged buildings abandoned or not rebuilt |
| `R_Equal_normalized` | Share rebuilt to an equal structure |
| `R_Higher_normalized` | Share rebuilt to an improved structure |
| `Any_rebuilt_normalized` | Share rebuilt to either |
| `ec_zip` | Economic connectedness (Social Capital Atlas) |
| `clustering_zip` | Social clustering |
| `support_ratio_zip` | Support ratio |
| `median_income` | Median household income |
| `log_income` | Natural log of `median_income` |

### `zip_damage_coverage.csv` -- figure 5, top row
Every surveyed ZIP code, including those with no observed damage. Supplies the
undamaged comparison group, which the table above cannot: it is restricted to
damaged ZIP codes.

| column | description |
|---|---|
| `event` | `Harvey`, `Irma` or `Michael` |
| `zip` | Five-digit ZIP code |
| `n_total` | Street-view locations observed |
| `n_damaged` | Of those, how many were damaged. Zero marks an undamaged ZIP code |
| `pct_damaged` | `n_damaged` as a percentage of `n_total` |

### `evacuation_by_damage_timeseries.csv` -- figure 5, top row
Share of users displaced from their pre-disaster home ZIP code on each night,
split by whether that ZIP code had observed damage. A user counts as displaced
from the first night their dominant nighttime ZIP code differs from home.

| column | description |
|---|---|
| `disaster` | Storm and year |
| `damage_grp` | `Damaged` or `Undamaged`, by the user's home ZIP code |
| `rel_day` | Day relative to landfall |
| `p` | Share displaced by this day, as a fraction |
| `n` | Users observed on this day |
| `lo`, `hi` | 95% confidence interval |

The figure plots these relative to the first day of the window, so the curves
show displacement accumulated over the event.
