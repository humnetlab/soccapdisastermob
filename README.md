# Social capital and disaster mobility

Analysis code for a study of social capital and post-disaster mobility across
five US hurricanes: Harvey (2017), Irma (2017), Florence (2018), Michael (2018)
and Imelda (2019).

## Setup

```bash
pip install numpy pandas matplotlib scipy statsmodels geopandas
```

## Running

Each script reads `data/` and writes to `plots/`. No arguments are needed, and
no raw data, network access or API keys.

```bash
python 03_processing_lbs.py               # figures S4, S5, S6, S7
python 04_rg_variation.py                 # figure 3, tables S2, S3
python 05_ec_evac_association.py          # figures S8, 4, S9
python 06_ec_rebuilding_association.py    # figure 5, tables S5, S6
```

Scripts are independent and can be run in any order. Useful flags:

```bash
python 03_processing_lbs.py --figures s7        # one figure only
python 04_rg_variation.py --skip-figure         # tables only
python 06_ec_rebuilding_association.py --skip-tables
```

Run any script with `--help` for the full option list.

## Outputs

| Figure | Script | Content |
|---|---|---|
| 3 | 04 | Radius of gyration around landfall, by storm intensity |
| 4 | 05 | Economic connectedness and evacuation |
| 5 | 06 | Economic connectedness and rebuilding |
| S4–S7 | 03 | Mobility panel characterisation |
| S8, S9 | 05 | Connectedness against income; per-storm evacuation panels |

| Table | Script | Content |
|---|---|---|
| S2, S3 | 04 | Post-landfall change in radius of gyration |
| S5, S6 | 06 | Recovery outcomes on income and connectedness |

Scripts `01_processing_cdr.py`, `02_compare_soccap_metrics.py` and
`07_census_tract_level_associations.py` are pending.

## Data

The mobility sources behind this study — call detail records and
location-based service smartphone traces — are proprietary and cannot be
redistributed. What ships instead are the aggregate tables the figures are
drawn from: counts, binned distributions, evaluated kernel densities and
area-level totals. None contain individual trajectories or per-person records.

| Directory | Used by | Contents |
|---|---|---|
| [`data/lbs_aggregates/`](data/lbs_aggregates/) | 03 | Mobility panel aggregates |
| [`data/rg_variation/`](data/rg_variation/) | 04 | Radius of gyration panel, ZIP intensity, boundaries |
| [`data/ec_evac/`](data/ec_evac/) | 05 | Connectedness, evacuation effects, destination densities |
| [`data/rebuilding/`](data/rebuilding/) | 06 | ZIP-level damage and recovery outcomes |
| [`data/social_capital/`](data/social_capital/) | 05 | Social Capital Atlas, shipped unmodified |
| [`data/hurdat2/`](data/hurdat2/) | — | HURDAT2 best-track source, for provenance |

Each directory has a README documenting its columns.

## Regenerating the aggregates

Only possible with access to the proprietary data. Every script takes
`--stage aggregate` for this; see `--help` for the required paths. Additional
dependencies:

```bash
pip install pyspark duckdb shapely h3 census google-cloud-bigquery contextily
```

Two notes. Pass `--spark-temp-dir` on a filesystem with room to spare, since
shuffle spill can be large. And avoid passing an API key on a command line you
are redirecting to a log.
