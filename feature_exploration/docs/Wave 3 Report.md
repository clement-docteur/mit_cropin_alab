# Feature Exploration – Wave 3 Report (Families E structure, G terrain/context, H external maps)

*Prepared by Nathan Chu (with Claude) · 6 November 2026 · Analyses are our own (Google Earth Engine + Python) unless a source is given · **[inference]** = our interpretation, not tested · Plan: `00 Workflow Plan.md` (v4.1) · Every number traces to `Run Log.md` and `results/wave3/` · Read with the Wave 1 and Wave 2 Reports*

---

## 1. Summary

- **Terrain and landscape context (Family G) do not separate the classes.** No feature passes the decision rule. Slope, distance to water, height above drainage, and patch size are weak, and several point the opposite way from the hypothesis.
- **Canopy structure (Family E) adds little that we can interpret.** Height uniformity (H12) and number of canopy layers (H14) are not supported. Height-for-age (H13) cannot be tested properly with our data. GEDI lidar has almost no shots at Cropin's points.
- **Public maps (Family H) look excellent on MapBiomas points but fail at Cropin's alerts.** The MapBiomas 10 m map falls from AUC 0.90 to 0.56. JRC's forest-type map labels 78% of MapBiomas plantations "planted" but only 12% of Cropin's QC plantations. We recommend using these maps as **benchmarks, not features**.
- **Safety finding across waves:** the land-history features from Wave 1 call **47% of disturbed-and-regrown natural forest "plantation"** (23 of 49 points), against 10% for the baseline. Seasonal (Wave 2) and spectral (Wave 1) features do not have this problem.

"Forest error" always means **real forest called plantation** (our dangerous error), not Cropin's "false positive" (a plantation harvest flagged as deforestation).

## 2. Data and method (short)

Same points, splits, evaluation, and decision rule as Waves 1–2 (see Wave 1 Report §2). The expected direction for each hypothesis was written in `Run Log.md` before any Wave 3 data were seen.

| Input | Hypotheses | Source |
|---|---|---|
| ETH global canopy height 2020 (10 m) and its model uncertainty | H12, H13, H17 | Lang et al. 2022, [arXiv:2204.08322](https://arxiv.org/abs/2204.08322); [GEE community catalog](https://gee-community-catalog.org/projects/canopy/) |
| GEDI lidar L2A/L2B monthly rasters (top height, cover, foliage height diversity), quality-filtered, 3 years ending at the cutoff | H12, H14 | [GEE catalog, L2A](https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI02_A_002_MONTHLY); [L2B](https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI02_B_002_MONTHLY) |
| ESA WorldCover 2021: land-cover shares within 250/500 m, tree-patch size and edge fraction | H29, H30, H36 | [GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200) |
| Copernicus DEM slope; JRC Global Surface Water; MERIT Hydro height above drainage | H31, H32 | [JRC GSW](https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater); [MERIT Hydro](https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1) (Yamazaki et al. 2019, [doi:10.1029/2019WR024873](https://doi.org/10.1029/2019WR024873)) |
| WDPA protected areas; WorldClim temperature and rainfall | H35, H38 | [WDPA](https://developers.google.com/earth-engine/datasets/catalog/WCMC_WDPA_current_polygons); [WorldClim](https://developers.google.com/earth-engine/datasets/catalog/WORLDCLIM_V1_BIO) |
| Sentinel-2 summer NDVI and SWIR at the point vs a 30–100 m ring | H37 | Our composite (Cloud Score+ ≥ 0.6) |
| MapBiomas 10 m map; JRC global forest types 2020 V1; NFW 2020; Hansen tree cover 2000 | External benchmarks | [JRC GFT](https://developers.google.com/earth-engine/datasets/catalog/JRC_GFC2020_subtypes_V1); [NFW](https://developers.google.com/earth-engine/datasets/catalog/projects_nature-trace_assets_forest_typology_natural_forest_2020_v1_0_collection) |

## 3. Results

### 3.1 Models

| Model | MapBiomas AUC | MapBiomas recall @1% / 2% / 5% | QC AUC [95% CI] | QC forest error at MapBiomas 5% threshold | QC recall |
|---|---|---|---|---|---|
| Baseline (SAG + height + height SD) | 0.841 | 19% / 27% / 42% | 0.782 [0.70–0.84] | 17.2% (10/58) | 64% |
| Baseline + E (ETH height, ETH model uncertainty) | 0.869 | 27% / 38% / 53% | 0.827 [0.76–0.88] | 15.5% (9/58) | 64% |
| Baseline + H (NFW, MapBiomas map, JRC "planted") | 0.972 | 67% / 82% / 90% | 0.786 [0.71–0.85] | 8.6% (5/58) | **46%** |
| Waves 1–2 preview (A, B, C, D, F) | 0.969 | 75% / 81% / 89% | 0.871 [0.82–0.91] | 12.1% (7/58) | 71% |
| Waves 1–2 preview + E | 0.974 | 74% / 81% / 90% | 0.887 [0.83–0.94] | 10.3% (6/58) | 70% |
| Waves 1–2 preview + H | 0.985 | 80% / 88% / 93% | 0.857 [0.81–0.91] | 10.3% (6/58) | 53% |

Family G has no model because no feature passed. Nested cross-validation gives the same MapBiomas AUCs (Baseline + E 0.869, + H 0.972).

### 3.2 Hypotheses tested (from *Task 1.2 – Hypotheses Master List*)

| Hypothesis | Expected | Result |
|---|---|---|
| H12: uniform canopy height | Less height spread in plantations | **Not supported:** ETH height spread is *higher* in plantations (AUC 0.67); GEDI ≈ 0.5. Meta canopy height gave the same answer in Phase 0 |
| H13: height for stand age | Plantations taller for their age | **Not testable:** 86% of points have no disturbance in 25 years, so stand age is unknown. The ratio we built mostly re-expressed "years since disturbance" and was dropped |
| H14: one canopy layer vs several | Lower foliage height diversity in plantations | **Not supported:** GEDI AUC ≤ 0.55; too few GEDI shots on QC to test |
| H17: height by forest type | No stated direction | Plantations are taller on average (ETH AUC 0.69 on MapBiomas, 0.74 on QC) |
| H29: stand shape | Compact, straight-edged plantation blocks | **Not testable:** 89% of points sit in tree patches larger than the 1,024-pixel limit |
| H30: stand size | Larger patches in plantations | **Not supported** (0.53, opposite direction) |
| H31: slope | Plantations on gentler slopes | Direction as expected but weak (0.59; QC 0.51) |
| H32: riparian position | Plantations farther from water | **Not supported** (≤ 0.52) |
| H35: protected areas | No stated direction | Weak (0.58; QC 0.50) |
| H36: neighbourhood composition | No stated direction | Weak (all ≤ 0.55) |
| H37: edge contrast | No stated direction | Weak (0.56; QC 0.50) |
| H38: climate and elevation | No stated direction (for stratifying) | Weak (≤ 0.62); kept for stratification only |

**Could not be tested with current access:**

| Hypothesis | Why not |
|---|---|
| H34 Distance to mills | No dataset |
| H39 CAR/SICAR | Not in GEE |
| H40 Planting registries | Not in GEE |
| H41 FSC certification | Not in GEE |
| H42 Municipal silviculture statistics | IBGE SIDRA table 5930 is blocked from our analysis environment; it could be downloaded by hand |
| H43 Forest-type map | Not in GEE |
| H10 Planting-year products, WRI planted-trees database | No verified GEE asset |

### 3.3 Public maps at Cropin's alerts

| Map | MapBiomas AUC | QC AUC | At the map's own class: MapBiomas recall / forest error | QC recall / forest error |
|---|---|---|---|---|
| MapBiomas 10 m map (plantation class) | 0.895 | 0.559 | 84% / 5% | 19% / 7% |
| JRC forest types 2020 ("planted") | 0.878 | 0.552 | 78% / 2% | 12% / 2% |
| NFW 2020 (low = plantation) | 0.957 | 0.726 | — | — |
| Hansen tree cover 2000 | 0.747 | 0.665 | — | — |

Two effects likely combine:

- **Circularity.** The MapBiomas map's AUC rises with how many MapBiomas interpreters agreed (0.70, 0.79, 0.94), which fits a map–label echo. JRC's forest-type map lists NFW among its inputs ([GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/JRC_GFC2020_subtypes_V1)).
- **Selection.** Cropin's QC points are, by design, places where Cropin's forest layer (built partly from these maps) said "forest", so the maps are selected to miss these plantations.

We can't separate the two effects. Either way, these maps add MapBiomas accuracy that does not carry over to Cropin's alerts. **[inference]** This also helps explain why small plantations end up in Cropin's forest layer in the first place.

## 4. Safety check: disturbed natural forest

Natural forest that was disturbed (for example by fire or logging) and then regrew can look like a harvested plantation. Cropin raised this case. We tested it on the 49 MapBiomas forest points with at least one disturbance in the last 25 years.

| Model | Forest called plantation (5% overall threshold) |
|---|---|
| Baseline | 10.2% (5/49) |
| Baseline + A (spectral) | 6.1% (3/49) |
| Baseline + B (seasonal) | 10.2% (5/49) |
| **Baseline + D (history, Wave 1)** | **46.9% (23/49)** |

The history features learn "was cleared before → plantation". Because these points are only 3.5% of the forest sample, the overall forest error (5%) hides this. The final model must either leave out the history features or guard against this case.

## 5. Checks and limits

- **Independent review:** a separate agent recomputed every model number to 4 decimals and rebuilt five features from the raw files (30 of 30 match). No code bugs. It raised three issues, all handled:
  - The height-for-age ratio was not a valid H13 test, so it was **dropped**.
  - The MapBiomas map year for 4 QC points was after their cutoff, so we **fixed** it. No result changed.
  - ETH model uncertainty is partly a stand-in for height, region, and cloudiness, though it keeps some signal within blocks (AUC 0.77). We treat it as uninterpretable.
- **Data losses:** GEDI's first export lost its main columns and was re-run. GEDI has no shots within 100 m for 56% of points, and for nearly all QC points (GEDI was switched off for much of 2023–24).
- **As-of:** WDPA uses current boundaries, which is a minor caveat; that feature is not used in any model.
- **Small, biased QC sample:** 58 forest points, all at places Cropin already flagged. **Cropin has no labels for forest it misses.**

## 6. Implications for Phase 4 (combining families)

| Family | Recommendation |
|---|---|
| A: Spectral, B: Seasonal | Core candidates: they help on Cropin's points and are safe on disturbed forest |
| E: Structure | Optional, tested as an add-on; little interpretability |
| D: History | Exclude, or include only with a guard against disturbed natural forest |
| C: Texture, F: Radar, G: Terrain/context | Exclude: no gain on Cropin's points |
| H: Public maps | Benchmarks only |

## 7. Files

- `results/wave3/`: screen, models, nested CV, selected features, feature tables
- `scripts/`: `w3_*.py` and the GEE scripts `w3_batch3_gee.js` and `w3b_gedi_gee.js`
