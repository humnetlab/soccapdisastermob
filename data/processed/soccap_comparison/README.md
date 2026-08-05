# Social capital comparison

Tables behind figure 2 and figure S3, produced by
`02_compare_soccap_metrics.py`. Every row describes a ZIP code or a distance
bin; no per-user or per-tie record is included.

### `zip_cdr_vs_facebook.csv` -- figure 2
CDR-derived support ratio and clustering per ZIP code (recomputed at the main
interaction threshold, m=3), alongside the Social Capital Atlas values for the
same ZIP codes.

| column | description |
|---|---|
| `zip` | Five-digit ZIP code, zero-padded |
| `n_edges` | Within-ZIP CDR edges backing `support_ratio_cdr` |
| `support_ratio_cdr` | CDR support ratio |
| `n_users` | CDR users with a home ZIP backing `clustering_cdr` |
| `clustering_cdr` | CDR clustering coefficient, averaged over the ZIP's residents |
| `support_ratio_fb` | Social Capital Atlas `support_ratio_zip` |
| `clustering_fb` | Social Capital Atlas `clustering_zip` |

Figure 2 is drawn on ZIP codes with at least 50 within-ZIP edges, so a handful
of interactions cannot dominate a ZIP's estimate.

### `distance_decay.csv` -- figure S3
Tie density against home distance, for the CDR network and for Meta's Social
Connectedness Index (SCI), each expressed relative to a geographic null (CDR:
random user pairs; SCI: the distribution of distances between all scored ZIP
pairs). A value above 1 means ties at that distance are over-represented
relative to how common the distance is geographically.

| column | description |
|---|---|
| `distance_km` | Log-spaced distance bin centre |
| `p_cdr_bay_area` | CDR tie density ratio, Bay Area users |
| `p_sci_us` | SCI density ratio, all US ZIP pairs |
| `p_sci_bay_area` | SCI density ratio, Bay Area ZIP pairs only |

Within-ZIP pairs use the square root of the ZIP code's area in place of a
zero centroid-to-centroid distance, so the shortest bins are not degenerate.
