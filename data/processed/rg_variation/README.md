# Radius of gyration

Tables behind figure 3 and tables S2/S3, produced by `04_rg_variation.py`. Each
row is a cell mean over thousands of user-days or an area-level count.

### `rg_intensity_panel.csv`
Mean radius of gyration per disaster, day, intensity bin and income group. Drives
both the line panels and the regressions.

| column | description |
|---|---|
| `disaster` | Storm and year |
| `t` | Day relative to landfall, -7 to +7 |
| `int_bin` | Wind intensity at the user's home ZIP code |
| `income_group` | `High` or `Low`, split at the median ZIP median income |
| `mean_rog` | Mean radius of gyration in km over the users in the cell |
| `n` | User-days contributing to the mean |

Bins are `No TS winds (<34kt)`, `Tropical-storm (34-49kt)`, `Gale-force
(50-63kt)` and `Hurricane-force (>=64kt)`. Imelda appears only in the first: no
ZIP code in its footprint reached tropical-storm force.

### `zip_intensity.csv`
Wind intensity per residential ZIP code; 9,120 rows. Assigns every user to an
intensity bin through their home ZIP code, so it underlies the panel above as
well as the maps.

| column | description |
|---|---|
| `disaster` | Storm and year |
| `home_zip` | Five-digit ZIP code, zero-padded |
| `int_bin` | Wind intensity bin, as above |

Derived from HURDAT2 best-track data, a copy of which is in
[`data/hurdat2/`](../hurdat2/): the 34-, 50- and 64-knot wind radii are buffered
along each track and dissolved, and each ZIP code takes the highest threshold
whose footprint contains it. Shipped as a fixed input rather than recomputed, so
the published figures reproduce exactly.

### `evacuation_rate_by_intensity.csv`
Evacuation rate by intensity, for the final panel of figure 3.

| column | description |
|---|---|
| `disaster` | Storm and year |
| `int_bin` | Wind intensity bin |
| `eligible` | Users with an identified pre-disaster home in the bin |
| `evacuated` | Of those, how many evacuated |
| `evacuation_rate` | `evacuated` as a percentage of `eligible` |

### `zcta_storm_footprints.gpkg`
ZIP boundaries for the maps, in EPSG:4269. Derived from the 2019 TIGER/Line ZCTA
shapefile, which is 814 MB nationwide: only the 4,100 ZIP codes appearing above
are kept, simplified to about 500 m, giving 3.3 MB. Indistinguishable from the
full shapefile at the scale the maps are drawn.

| column | description |
|---|---|
| `home_zip` | Five-digit ZIP code |
| `geometry` | Simplified boundary |
