# Harvest-cycle inputs (public data)

Inputs for `harvest_cycles/harvest_cycles_findings.ipynb`, extracted once from public satellite archives at the
team's 1,831 MapBiomas validation points and at the 433 Cropin QC clearings.

| File | Contents |
|---|---|
| `points.csv` | One row per point: `pid` (`V…` MapBiomas, `Q…` Cropin QC), location, MapBiomas label (1 = plantation, 0 = natural forest), municipality code, and the area of each QC clearing. **QC labels are left empty**: they are confidential and the notebook reads them from Cropin's GeoJSON in `CROPIN-MIT-DATA/cropin_qc_data/`. `Q000`, `Q001`, … follow the order of the polygons in that file; QC locations are each polygon's representative point. |
| `series_nbr_median.csv.gz` | Yearly NBR per point, 1985–2020: median of all clear observations in the year |
| `series_nbr_dry_median.csv.gz`, `series_nbr_wet_median.csv.gz`, `series_nbr_max.csv.gz` | The same with a dry-season (Apr–Sep) median, a wet-season (Oct–Mar) median, or the yearly maximum |
| `series_ndvi_median.csv.gz` | Yearly NDVI per point (median), for comparison |
| `hansen_gfc.csv` | Hansen Global Forest Change v1.12 at each point: tree cover 2000, gain 2000–2012, loss year, and the number of 3×3 neighbours with a loss up to 2020 |
| `canopy_height.csv` | Meta / WRI global canopy height v1 (about 1 m; at these points imagery from 2010–2019, three quarters 2018–2019) at each point: mean over an 11 × 11-pixel window, about 12 m × 12 m (`chm_mean`), and the centre pixel (`chm_centre`), in metres |
| `val_label_history.csv` | MapBiomas reference class per year, 1985–2022, at the MapBiomas points (3 = forest, 9 = plantation) |

## How they were made
- **Landsat:** Landsat 4/5/7/8 Collection 1 Level-1 top-of-atmosphere reflectance from Google's public archive
  (`gcp-public-data-landsat`): 8,063 low-cloud scenes (about 6 per path/row and year), read at each point's pixel,
  cloud- and shadow-masked with the quality band. NBR = (NIR − SWIR2) / (NIR + SWIR2); Landsat 4/5 and 8 are shifted
  onto the Landsat 7 scale with offsets measured where two sensors saw the same point within 8 days. A median of
  6 clear observations per point and year; 0.4% of point-years have none (left empty).
- **Hansen:** tiles of `earthenginepartners-hansen/GFC-2024-v1.12` on Google Cloud Storage; losses after 2020 are
  ignored by the notebook.
- **MapBiomas:** the MapBiomas validation points (`mapbiomas_85k_points_validation`), restricted to the rows of the
  team's `data_exploration/cropin_model_features_2022.csv`.
- Only imagery up to 31 December 2020 is used (EUDR cutoff).
- **Canopy height:** tiles of `forests/v1/alsgedi_global_v6_float/chm/` on the public `dataforgood-fb-data` bucket
  (the dataset Cropin's forest layer uses through Earth Engine).

## Optional: Cropin's exact layers
`harvest_cycles/gee/cropin_layers_at_points.js` (Earth Engine) writes `cropin_forest_layer.csv` (the parts of Cropin's
forest layer at each point) and `cropin_sag_observations.csv` (every SAG value at each point, Oct–Jan 2019–2025).
Saved here, they replace the stand-ins in section 8 of the notebook. Neither holds Cropin's labels.
