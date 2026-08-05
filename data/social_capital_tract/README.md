# Reconstructed tract-level social capital

`reconstructed_tract_social_capital.csv` is a reconstruction of social capital
measures at US census tract level, covering 72,166 tracts. It is shipped exactly
as produced, unmodified, so that readers can see it is used as published.

Scripts 05 and 06 measure social capital with the Social Capital Atlas, which is
published at ZIP code level. Script 07 repeats those analyses with this dataset,
which is native to tracts, as a check on a finer and independently constructed
geography.

| column | description |
|---|---|
| `census_tract_GEOID` | 11-digit census tract identifier |
| `EC_main` | Economic connectedness |
| `SR_main` | Support ratio |
| `VR_main` | Volunteering rate |
| `EC_80`, `SR_80`, `VR_80` | Same measures under an 80% reconstruction threshold |
| `EC_90`, `SR_90`, `VR_90` | Same, at 90% |
| `EC_100`, `SR_100`, `VR_100` | Same, at 100% |

The analyses use the `_main` columns. The threshold variants are retained for
anyone wishing to check sensitivity to the reconstruction cutoff.
