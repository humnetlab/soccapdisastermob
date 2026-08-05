# CDR aggregates

Tables behind figures S1 and S2, produced by `01_processing_cdr.py`. The raw
call detail records (CDR) are proprietary; these aggregates are counts, binned
distributions and evaluated densities, with no individual call records, home
locations, or per-person network ties.

### `s1_edges_by_threshold.csv` -- figure S1A
Network size at each interaction threshold `m`.

| column | description |
|---|---|
| `threshold_m` | Minimum combined call+SMS interactions for an edge to count |
| `n_edges` | Edges at that threshold |
| `n_nodes` | Distinct users with at least one surviving edge |

### `s1_degree_distribution.csv` -- figure S1B
Degree distribution `p(k)` of the network at the main threshold (m=3), binned
at unit width over 0-100.

| column | description |
|---|---|
| `degree` | Bin centre |
| `density` | Density at that degree |

### `s1_expansion_factor_kde.csv` -- figure S1C
ZIP-level expansion factor (ACS population divided by CDR-detected users),
evaluated as a kernel density on a fixed grid capped at the 99th percentile.

| column | description |
|---|---|
| `expansion_factor` | Grid point |
| `density` | Density at that point |

### `s1_expansion_factor_by_zip.csv` -- figures S1C, S1F
The expansion factor per ZIP code, feeding both the density above and the map.

| column | description |
|---|---|
| `zip` | Five-digit ZIP code, zero-padded |
| `expansion_factor` | ACS population / CDR-detected users in the ZIP |

### `bay_area_zctas.gpkg` -- figure S1F
ZCTA boundaries for the Bay Area map, in EPSG:4326. Derived from the 2019
TIGER/Line ZCTA shapefile, restricted to the ZIP codes appearing in the
expansion factor table above and simplified to keep the file small.

| column | description |
|---|---|
| `zip` | Five-digit ZIP code |
| `geometry` | Simplified boundary |

### `s1_homes_by_threshold.csv` -- figure S1D
Sensitivity of the home-location count to the night-events threshold.

| column | description |
|---|---|
| `threshold_events` | Minimum nighttime voice events at a user's top cell |
| `n_homes` | Users whose top cell reaches that threshold |

### `s1_unique_locations_distribution.csv` -- figure S1E
Distribution of the number of distinct cells visited per user, `p(S)`, binned
at width 2 over 0-100.

| column | description |
|---|---|
| `locations` | Bin centre |
| `density` | Density at that bin |

### `s2_strength_by_threshold.csv` -- figure S2A
Degree distribution `p(s)` at each interaction threshold swept in figure S2.

| column | description |
|---|---|
| `threshold_m` | Interaction threshold |
| `degree` | Degree value |
| `p` | Share of users at that degree, for this threshold |

### `s2_soccap_kde_by_threshold.csv` -- figure S2B-D
Distribution of clustering, support ratio and economic connectedness at each
interaction threshold, evaluated as kernel densities on fixed grids.

| column | description |
|---|---|
| `threshold_m` | Interaction threshold |
| `metric` | `clust`, `sr` or `ec` |
| `x` | Grid point, on the metric's own scale |
| `density` | Density at that point |
