# HURDAT2

`hurdat2-1851-2024-040425.txt` is the Atlantic hurricane database published by
the NOAA National Hurricane Center, covering 1851-2024 as revised 2025-04-04.
Public-domain US government data, included for provenance.

Upstream: <https://www.nhc.noaa.gov/data/#hurdat>

Each storm appears as a header line giving its identifier, name and record
count, followed by best-track observations. Each observation gives date, time,
status, position, maximum sustained wind, minimum pressure, and the radii within
which 34-, 50- and 64-knot winds were observed in each quadrant. Missing values
are `-999`.

Those wind radii are buffered along each track and dissolved, and every
residential ZIP code takes the highest threshold whose footprint contains it:

| Bin | Maximum sustained wind |
|---|---|
| No TS winds | < 34 kt |
| Tropical-storm force | 34-49 kt |
| Gale force | 50-63 kt |
| Hurricane force | >= 64 kt |

The result ships as
[`data/processed/rg_variation/zip_intensity.csv`](../processed/rg_variation/zip_intensity.csv), so
no script parses this file at run time.
