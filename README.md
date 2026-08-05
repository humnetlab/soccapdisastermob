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
python 01_processing_cdr.py                     # figures S1, S2
python 02_compare_soccap_metrics.py              # figure 2, figure S3
python 03_processing_lbs.py                     # figures S4, S5, S6, S7
python 04_rg_variation.py                       # figure 3, tables S2, S3
python 05_ec_evac_association.py                # figures S8, 4, S9
python 06_ec_rebuilding_association.py          # figure 5, tables S5, S6
python 07_census_tract_level_associations.py    # figures S10-S12, table S7
```

Scripts are independent and can be run in any order. Useful flags:

```bash
python 01_processing_cdr.py --figures s1        # one figure only
python 03_processing_lbs.py --figures s7        # one figure only
python 04_rg_variation.py --skip-figure         # tables only
python 06_ec_rebuilding_association.py --skip-tables
```

Run any script with `--help` for the full option list.

## Outputs

| Figure | Script | Content |
|---|---|---|
| 2 | 02 | CDR-derived support ratio and clustering against the Social Capital Atlas |
| 3 | 04 | Radius of gyration around landfall, by storm intensity |
| 4 | 05 | Economic connectedness and evacuation |
| 5 | 06 | Economic connectedness and rebuilding |
| S1, S2 | 01 | CDR network construction and metric sensitivity to the interaction threshold |
| S3 | 02 | Tie probability against home distance, CDR vs. Meta SCI |
| S4–S7 | 03 | Mobility panel characterisation |
| S8, S9 | 05 | Connectedness against income; per-storm evacuation panels |
| S10–S12 | 07 | The same associations at census tract level |

| Table | Script | Content |
|---|---|---|
| S2, S3 | 04 | Post-landfall change in radius of gyration |
| S5, S6 | 06 | Recovery outcomes on income and connectedness |
| S7 | 07 | Recovery outcomes at census tract level |

Note: table S2 and S3 (04) share their numbers with figure S2 and S3 (02) --
the paper's supplementary tables and figures are numbered in independent
sequences.

Scripts 01 and 02 build the CDR-derived social network and compare its social
capital metrics against the Social Capital Atlas. Scripts 05 and 06 then
measure social capital with the Atlas itself, which is published at ZIP code
level; script 07 repeats both analyses with a reconstructed dataset native to
census tracts, as a check on a finer and independently constructed geography.

## Data

The mobility sources behind this study — call detail records and
location-based service smartphone traces — are proprietary and cannot be
redistributed. What ships instead are the aggregate tables the figures are
drawn from: counts, binned distributions, evaluated kernel densities and
area-level totals. None contain individual trajectories or per-person records.

| Directory | Used by | Contents |
|---|---|---|
| [`data/processed/cdr_aggregates/`](data/processed/cdr_aggregates/) | 01 | CDR network construction and metric sensitivity aggregates |
| [`data/processed/soccap_comparison/`](data/processed/soccap_comparison/) | 02 | CDR-vs-Atlas ZIP metrics, tie-distance decay |
| [`data/processed/lbs_aggregates/`](data/processed/lbs_aggregates/) | 03 | Mobility panel aggregates |
| [`data/processed/rg_variation/`](data/processed/rg_variation/) | 04 | Radius of gyration panel, ZIP intensity, boundaries |
| [`data/processed/ec_evac/`](data/processed/ec_evac/) | 05 | Connectedness, evacuation effects, destination densities |
| [`data/processed/rebuilding/`](data/processed/rebuilding/) | 06 | ZIP-level damage and recovery outcomes |
| [`data/social_capital/`](data/social_capital/) | 05 | Social Capital Atlas, shipped unmodified |
| [`data/processed/tract_associations/`](data/processed/tract_associations/) | 07 | Tract-level connectedness, evacuation and recovery |
| [`data/social_capital_tract/`](data/social_capital_tract/) | 07 | Reconstructed tract social capital, shipped unmodified |
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
