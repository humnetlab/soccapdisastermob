# LBS aggregates

Tables behind figures S4-S7, produced by `03_processing_lbs.py`. The raw
location-based service traces are proprietary; these aggregates are counts,
binned distributions and evaluated densities, with no individual trajectories
or per-person records.

### `s4_user_activity_levels.csv` -- figure S4
Users at each combination of observation span and stay-point count.

| column | description |
|---|---|
| `duration_days` | Days between a user's first and last stay point |
| `num_stays` | Total stay points recorded |
| `count` | Users with this (span, count) pair |

### `s5_mobility_distributions.csv` -- figure S5
Three mobility distributions per disaster, in long form.

| column | description |
|---|---|
| `disaster` | `Harvey`, `Irma`, `Imelda` or `Michael & Florence` |
| `quantity` | `departure_time`, `stay_duration` or `location_rank` |
| `x` | Hour of day, duration bin centre in hours, or rank |
| `y` | Density, or mean visitation frequency for `location_rank` |

### `s6_census_vs_smartphone.csv` -- figure S6
Tract population against detected users. GEOIDs are omitted so the table cannot
be used to derive smartphone penetration for a named tract.

| column | description |
|---|---|
| `disaster` | Disaster group, as above |
| `population` | ACS 2019 five-year tract population (B01003_001E) |
| `mobile_count` | Distinct users with a detected home in the tract |

### `s7_threshold_summary.csv` -- figure S7A
Sensitivity of the evacuation measure to the nighttime dwell threshold.

| column | description |
|---|---|
| `disaster` | Storm and year |
| `threshold_minutes` | Nighttime dwell threshold |
| `n_users_with_pre_home` | Users with an identified pre-disaster home |
| `n_users_with_disaster_week_location` | Users located during the disaster week |
| `n_evacuated` | Users whose disaster-week location differed from home |
| `evacuation_rate` | `n_evacuated` as a share of users with a pre-disaster home |

### `s7_evacuation_distance_kde.csv` -- figure S7B
Evacuation distances at the 480-minute threshold, published as kernel densities
on a fixed 0-200 km grid rather than per-evacuee values.

| column | description |
|---|---|
| `disaster` | Storm and year |
| `distance_km` | Grid point |
| `density` | Density at that distance |
