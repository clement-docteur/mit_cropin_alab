# Feature Exploration – Summary of Findings (Phases 0–5)

*Prepared by Nathan Chu (with Claude) · 8 November 2026 · Analyses are our own (Google Earth Engine + Python) unless a source is given · **[inference]** = our interpretation, not tested · Detailed evidence: `Wave 1 Report`, `Wave 2 Report`, `Wave 3 Report`, `Run Log.md`, and `results/` in the `Feature Exploration` folder · Method: `00 Workflow Plan.md`*

---

## 1. Summary

1. **Four simple features beat the baseline clearly.** They are Cropin's SAG colour signal measured in spring, the timing of the yearly colour cycle, summer canopy moisture, and late-spring greenness. On Cropin's QC points they lower forest error from about 17% to about 7% at the same plantation recall (QC AUC 0.78 → 0.85). On MapBiomas points they nearly double plantation recall at 5% forest error (42% → 72%).
2. **Cropin's SAG works better in spring.** SAG measured over September–November separates plantations from natural forest better than the December–January window Cropin uses (AUC 0.88 vs 0.83 on MapBiomas points; equal on Cropin's points).
3. **Most features that look strong on MapBiomas points do not carry over to Cropin's alerts.** This is true of land history, texture, and the public maps. The public maps (MapBiomas, JRC forest types, NFW) miss most of the plantations at Cropin's alerts.
4. **Land-history features are dangerous on their own.** They call 47% of disturbed-and-regrown natural forest "plantation", which is the case Cropin warned about.
5. **The models rank stands consistently across states, years, and cloud levels, but the decision threshold drifts in unusual years.** The final layer should use multi-year (3-year) windows.
6. **Of the 43 numbered hypotheses:** 10 are supported (most only on MapBiomas points), 11 are not supported, 7 are weak, 1 is context-dependent, 9 could not be tested with our data and access, 4 are not yet tested, and 1 (H17) is descriptive.

Throughout, **"forest error" means real forest called plantation**. That is our dangerous error, not Cropin's "false positive" (a plantation harvest flagged as deforestation). **Cropin has no labels for forest it misses**, so no number in this document shows the overall forest error.

## 2. The question and how we judge answers

| Item | Detail |
|---|---|
| Goal | For pixels with trees in Paraná, Santa Catarina, and Rio Grande do Sul, give the probability of **plantation vs natural forest** at 10 m, so Cropin can drop false deforestation alerts caused by plantation harvests |
| Legal definitions | EUDR: plantation = planted, 1–2 species, even age, regular spacing; naturally regenerating forest otherwise ([Regulation (EU) 2023/1115, Art. 2](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX%3A02023R1115-20251226)). Team decisions: wild pine, bracatinga, restoration plantings, shade-grown erva-mate, and abandoned plantations = forest; planted araucaria and black wattle = plantation (*Task 1.2 – Hypotheses Master List*, §1) |
| Error priority | Missing real forest is the worst error. Cropin's tolerance is "technically zero" (Cropin meeting transcript, 14 Oct 2026). We always report **plantation recall and forest error together**, at 1%, 2%, and 5% forest error |
| Main score | AUC (0.5 = coin flip, 1.0 = perfect ranking), with 95% confidence intervals from a spatial block bootstrap |

## 3. Data

| Set | What it is | Size | Use |
|---|---|---|---|
| MapBiomas validation points ([MapBiomas](https://brasil.mapbiomas.org/en/?p=951)) | Points labelled yearly by 3 interpreters. We kept points labelled forest (class 3) or plantation (class 9) in **every** year 2013–2022 | 1,817 dev points (1,406 forest, 411 plantation); 1,753 after removing points without tree cover | Training and cross-validation |
| MapBiomas **lockbox** | 20% of 0.5° blocks, chosen at random by state | 436 points | **Sealed** until the final test |
| Cropin QC points | Small areas at recent clearings, labelled forest or plantation by Cropin from historical Google Earth imagery | 415 after removing near-duplicates; dev half 207 (58 forest, 149 plantation) | Checking that results carry over to real alerts. **Never used for training** |
| QC **held-back half** | The other half, split by 0.25° blocks | 208 (59 forest, 149 plantation) | **Sealed** until the final test |

**Clearing dates:** Cropin's QC file has no clearing dates, so we estimated them from monthly Sentinel-2 greenness. 91% of the clearings fall in April–September 2026.

## 4. How every hypothesis was tested

The same protocol was used for every feature family (full rules in `00 Workflow Plan.md`):

1. **As-of rule.** Features use only data from before each point's clearing (MapBiomas: up to 2022; QC: up to the year before the estimated clearing). Time-series features use the 3 years ending at that cutoff.
2. **Spatially blocked cross-validation.** Points are grouped in 0.5° blocks, and a block is never split between training and testing. This stops nearby, similar points from inflating scores.
3. **Decision rule for each feature.** A feature passes only if all of these hold:
   - It is significant after correction for testing many features (Benjamini–Hochberg, $q < 0.05$).
   - The lower end of its AUC confidence interval is above 0.60.
   - It points the same way in all three states and the way the hypothesis predicted.
   - It adds at least 5 percentage points of plantation recall at 5% forest error over the baseline, and does not get worse at 1% or 2%.
4. **Baseline.** Cropin's SAG + Meta canopy height + height spread. The NFW forest map is left out of the baseline, because its training labels may resemble MapBiomas labels (unverified).
5. **Expected directions were written down before seeing the data** for Waves 2 and 3 (`Run Log.md`).
6. **Threshold-transfer check.** The threshold that gives 5% forest error on MapBiomas is applied unchanged to Cropin's points, and we report the forest error there.
7. **Interpretable models only.** Logistic regression with at most ~10 features, chosen by stability selection, and a depth-3 decision tree for comparison.
8. **Independent review.** After each wave, a separate AI agent was given the code and data (but not our conclusions) and recomputed the key numbers. Every reported number was reproduced. The reviews found 8 issues, all fixed or documented (`Run Log.md`).

## 5. Hypotheses tested and results

AUC is shown as **MapBiomas / Cropin QC**, for the single best feature for each hypothesis. Hypothesis numbers follow *Task 1.2 – Hypotheses Master List*.

### 5.1 Spectral and colour (Wave 1, Family A)

| Hypothesis | Feature tested | AUC | Verdict |
|---|---|---|---|
| Cropin's SAG (dark-green plantations) | SAG angle from blue, green, and red reflectance, Oct–Jan composite | 0.85 / 0.79 | **Supported**: strongest single signal |
| H20 pine vs broadleaf spectra | Shortwave infrared (B11) reflectance | 0.71 / 0.73 | Supported, but adds nothing once SAG is included |
| H19 eucalyptus red-edge | Red-edge index (NDRE) | 0.55 / 0.63 | Weak |
| (exploratory) Canopy moisture | Summer NDMI (plantations slightly wetter) | 0.68 / 0.70 | **Supported**: in the final model |

### 5.2 Texture and uniformity (Wave 1, Family C)

| Hypothesis | Feature tested | AUC | Verdict |
|---|---|---|---|
| H18 lower image texture | Grey-level texture (GLCM) at 3 window sizes | 0.71 / 0.44 | **Context-dependent**: direction flips at Cropin's alerts |
| H24 uniform stands | Spread within SNIC segments | ≤ 0.60 | Not supported; also the most expensive feature tried |
| H16 planting rows | Directional texture | 0.59 / 0.57 | Not supported: rows of ~3 m are below what 10 m pixels can show **[inference]** |

This matches Deepak's caution about texture at 10 m (Cropin meeting transcript, 21 Oct 2026, ~00:34:21–00:35:35).

### 5.3 Land history (Wave 1, Family D)

Built from yearly Landsat data since 1985, plus LandTrendr and CCDC (the two methods Cropin has tried) and Hansen forest loss.

| Hypothesis | Feature tested | AUC | Verdict |
|---|---|---|---|
| H1 earlier clear-cuts | Number of disturbance events | 0.77 / 0.54 | Supported on MapBiomas, weak at alerts |
| H2 regular rotation | Mean years between events | 0.60 / 0.50 | Not supported |
| H3 regrowth speed | NDVI gain in the 3 years after a disturbance | 0.79 / 0.78 | Supported, but only measurable after a disturbance |
| H4 continuous forest | Years since NDVI was last below 0.6 | 0.85 / 0.56 | Supported on MapBiomas, weak at alerts |
| H5 sudden loss | Largest one-year NBR drop | 0.88 / 0.67 | Supported |
| H6 neighbours cleared together | Hansen loss within 150 m in the same year | 0.79 / 0.53 | Supported on MapBiomas only |
| H7 land before forest | NDVI in 1985–89 | 0.70 / 0.64 | Supported |
| H8 ground after harvest | Lowest NBR after the last event | 0.73 / 0.82 | Supported, after a disturbance only |
| H11 year-to-year stability | NDVI spread over 10 years | 0.77 / 0.64 | Supported |
| Cropin's CCDC and LandTrendr | Number of CCDC breaks; LandTrendr disturbance size | 0.88 / 0.61; 0.69 / 0.53 | Strong on MapBiomas, weak at alerts |
| H9 growth curve | — | — | Not tested directly (covered by H3) |

**Why history drops at Cropin's alerts:** 42% of QC plantations show no clearing in the last 25 years, against 14% of MapBiomas plantations. Most of Cropin's plantation alerts are **first harvests of long-standing stands**, which have no earlier clear-cut to detect.

### 5.4 Seasonal behaviour (Wave 2, Family B)

Built from 36 monthly Sentinel-2 composites per point.

| Hypothesis | Expected | Feature tested | AUC | Verdict |
|---|---|---|---|---|
| SAG by season | — | SAG in Sep–Nov vs Dec–Jan (Cropin's window) | **0.88 / 0.81** vs 0.83 / 0.81 | **Spring window is better** |
| H22 timing of the seasonal cycle | No stated direction | Phase of the yearly cycle (SAG, NDVI, red-edge) | 0.85 / 0.64 (SAG); 0.73 / 0.76 (NDVI) | **Supported, carries over** |
| H21 size of the seasonal swing | Plantations swing less | Amplitude of the yearly NDVI cycle | 0.58 / 0.53 | **Not supported** (opposite direction) |
| H23 drought response | Plantations stay greener | NDVI anomaly in drought months (CHIRPS rainfall) | 0.54 / — | **Not supported** (opposite direction); untestable on QC (almost no drought in 2023–25) |
| H27 surface temperature | No stated direction | Landsat summer and winter temperature | ≤ 0.59 | Not supported |

**[inference]** Spring is when native trees flush new leaves, which may be why colour differs most then. This has not been checked against phenology studies.

### 5.5 Radar (Wave 2, Family F)

| Hypothesis | Feature tested | AUC | Verdict |
|---|---|---|---|
| H25 Sentinel-1 C-band | VH−VV backscatter ratio | 0.77 / 0.67 | Partly supported; no gain at alerts once the baseline is included |
| H26 ALOS PALSAR L-band | HH, HV backscatter | ≤ 0.58 | Not supported |

### 5.6 Canopy structure (Wave 3, Family E)

| Hypothesis | Expected | Feature tested | AUC | Verdict |
|---|---|---|---|---|
| H12 uniform height | Less height spread in plantations | Height spread within 15/50 m (ETH, Meta, GEDI) | 0.67 / 0.67 | **Not supported**: plantations vary *more* |
| H13 height for age | Plantations taller for their age | Height ÷ years since disturbance | — | **Not testable**: 86% of points have no known age |
| H14 one canopy layer | Fewer layers in plantations | GEDI foliage height diversity | ≤ 0.55 | Not supported; almost no GEDI shots at QC points |
| H17 height by forest type | No stated direction | ETH canopy height 2020 | 0.69 / 0.74 | Plantations are taller on average |

### 5.7 Terrain and landscape context (Wave 3, Family G)

| Hypothesis | Feature tested | AUC | Verdict |
|---|---|---|---|
| H31 slope | Slope (Copernicus DEM) | 0.59 / 0.51 | Weak |
| H32 riparian position | Distance to water; height above drainage | ≤ 0.52 | Not supported |
| H30 stand size | Tree-patch size (WorldCover) | 0.53 | Not supported (opposite direction) |
| H29 stand shape | Patch edge fraction | — | Not testable: 89% of patches exceed the 1,024-pixel limit |
| H35–H38 protected areas, land-cover mix, edge contrast, climate | Distances, shares within 250/500 m, NDVI vs surrounding ring, WorldClim | ≤ 0.62 | Weak |

No terrain or context feature passed the decision rule.

### 5.8 Public maps (Wave 3, Family H; used as benchmarks)

| Map | AUC | Plantations it catches: MapBiomas / QC | Verdict |
|---|---|---|---|
| MapBiomas 10 m map | 0.90 / **0.56** | 84% / **19%** | Fails at Cropin's alerts |
| JRC global forest types 2020 ("planted") ([GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/JRC_GFC2020_subtypes_V1)) | 0.88 / **0.55** | 78% / **12%** | Fails at Cropin's alerts |
| Natural Forests of the World 2020 ([GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/projects_nature-trace_assets_forest_typology_natural_forest_2020_v1_0_collection)) | 0.96 / 0.73 | — | Strong on MapBiomas, much weaker at alerts |
| Hansen tree cover 2000 | 0.75 / 0.67 | — | Moderate |

Two effects probably combine:

- **Circularity.** These maps learned from labels like MapBiomas's, so they agree with MapBiomas points.
- **Selection.** Cropin's alerts are, by design, places where Cropin's forest layer (built partly from these maps) said "forest".

**[inference]** This helps explain why small plantations end up in Cropin's forest layer.

### 5.9 Not testable with current data or access

| Hypothesis | Why not |
|---|---|
| H10 planting-year products, H43 forest-type map | No verified dataset in Google Earth Engine |
| H34 distance to mills | No dataset |
| H39 CAR/SICAR, H40 planting registries, H41 FSC | Not in Google Earth Engine |
| H42 municipal silviculture statistics | IBGE table 5930 is blocked from our analysis environment; it could be downloaded by hand |

### 5.10 Still to do

| Item | Phase |
|---|---|
| H28 AlphaEarth embeddings (black-box benchmark) | Phase 7 |
| M1–M6 (where MapBiomas fails) | Phase 6 |
| H15 canopy gaps, H33 internal roads | Not tested separately |

## 6. Models: how much the features improve results

Dev data; the final, independent test is still to come (Phase 7).

| Model | MapBiomas AUC | MapBiomas plantation recall at 5% forest error | QC AUC [95% CI] | QC forest error at the MapBiomas threshold | QC recall |
|---|---|---|---|---|---|
| Baseline (SAG + height + height spread) | 0.84 | 42% | 0.78 [0.70–0.84] | 17.2% (10/58) | 64% |
| Baseline + NFW (comparison) | 0.96 | 83% | 0.78 | 13.8% | 43% |
| + A: spectral | 0.89 | 67% | 0.83 | 17.2% | 74% |
| + B: seasonal | 0.92 | 71% | 0.85 | **6.9%** | 59% |
| + C: texture | 0.85 | 50% | 0.79 | 17.2% | 60% |
| + D: history | 0.96 | 82% | 0.80 | 10.3% | 56% |
| + E: structure | 0.87 | 53% | 0.83 | 15.5% | 64% |
| + F: radar | 0.87 | 59% | 0.78 | 15.5% | 57% |
| + H: public maps | 0.97 | 90% | 0.79 | 8.6% | **46%** |
| **Lean 4** (SAG Sep–Nov, SAG phase, summer NDMI, Oct–Jan NDVI) | **0.91** | **72%** | **0.85** [0.81–0.89] | **6.9%** (4/58) | **66%** |
| P4 (10 spectral and seasonal features) | 0.93 | 73% | 0.87 [0.83–0.92] | 6.9% (4/58) | 67% |

- **Terrain/context (G)** has no row because no feature passed the decision rule.
- **Calibration:** with a threshold calibrated on MapBiomas points that resemble alert locations, both Lean 4 and P4 reach **5.2% forest error on QC** (3/58), at 60–63% recall.
- **Selection check:** a nested cross-validation, with feature selection repeated inside each fold, gives the same scores (P4: 0.923 vs 0.925), so selecting features did not inflate them.

**How Lean 4 reads:** a stand is more likely a plantation when it has these four signals:

1. **SAG in spring (Sep–Nov):** darker green.
2. **SAG phase:** the yearly colour cycle peaks at a different time from native forest.
3. **Summer NDMI:** slightly wetter in summer.
4. **Oct–Jan NDVI:** slightly less green in late spring.

All four point the same way as when tested alone.

## 7. Safety and robustness checks

### 7.1 Disturbed natural forest

Natural forest that was disturbed (for example by fire or logging) and then regrew can look like a harvested plantation, as Cropin warned. We tested the 49 MapBiomas forest points with a disturbance in the last 25 years.

| Model | Forest called plantation |
|---|---|
| Baseline | 10.2% (5/49) |
| Baseline + A: spectral | 6.1% (3/49) |
| Baseline + B: seasonal | 10.2% (5/49) |
| **Baseline + D: history** | **46.9% (23/49)** |
| Lean 4 / P4 | 10.2% / 12.2% |

Land history was therefore **left out of the final candidates**. **[inference]** Bracatinga stands (natural, cut and regrown on ~7-year cycles) are likely to be affected the same way.

### 7.2 Robustness (Phase 5)

| Test | Lean 4 | P4 |
|---|---|---|
| Leave one state out: forest error on held-out state | PR 4.9%, RS 4.3%, SC 6.7% | PR 3.9%, RS 2.8%, SC 7.9% |
| Train on Pampa only, test on Atlantic Forest | 13.9% | 8.5% |
| Few vs many clear Sentinel-2 images | 6.0% vs 4.1% | 7.0% vs 2.7% |
| QC, single-year features: 2023 / 2024 (flood year) / 2025 | 5.3% / **23.2%** / 14.5% | 7.0% / **16.1%** / 16.4% |
| QC, 3-year features | 3.4% | 6.9% |

- **Stable:** the ranking (AUC 0.86–0.94) holds across states, years, and cloud levels.
- **Not stable:** the threshold. It shifts in very wet years (2024 rainfall was 111% of normal) and when set on a small region.
- **[inference]** The layer should use 3-year windows, and thresholds should be set on the whole region and re-checked each year.

## 8. Limits

- **Small, biased QC sample.** 58 forest points in 30 blocks. A forest error of 4/58 has a 95% CI of about 3–17%. QC forest is only forest where Cropin already detected a clearing, and **Cropin has no labels for forest it misses**.
- **QC dev is no longer fully independent.** It informed several choices (flags, alert-like weights, model comparison). The sealed QC half and the MapBiomas lockbox are the independent test.
- **Labels.** MapBiomas labels come from 30 m Landsat interpretation. Points where only 1 of 3 interpreters agreed were left out of training and reported separately. QC labels have not been visually re-checked by us.
- **Different years.** MapBiomas features come from 2020–22 (dry: 75–97% of normal rainfall) and QC features from 2023–25 (wet: 100–114%). Some differences between the sets may be weather.
- **Region.** All findings are for southern Brazil. SAG timing and phase in particular depend on local phenology, and should be retested before a global layer.

## 9. Next steps

| Step | Status |
|---|---|
| Phase 6: error breakdown, including where MapBiomas fails (M1–M6) | Ready to start |
| Phase 7: one final test on the sealed QC half and MapBiomas lockbox, with AlphaEarth and public-map benchmarks | Needs a decision: Lean 4, P4, or both, committed before opening the sealed data |
| Visual check of the few QC forest points still called plantation (two labellers) | Optional; tells us whether they are label errors |
| Inputs that would help | QC clearing dates from Cropin; IBGE table 5930 (H42); Cropin scoring our layer on its blind holdout |
| For Cropin now | Test a **Sep–Nov SAG window** alongside Dec–Jan; it is a cheap change to an existing method |

## 10. Files

All in `Cropin - A Lab/Feature Exploration/`:

- `00 Workflow Plan` (method), `Run Log.md` (every run, decision, and review)
- `Wave 1 Report`, `Wave 2 Report`, `Wave 3 Report` (detailed results)
- `results/` (tables), `scripts/` (all code, including Google Earth Engine scripts)
- Hypothesis verdicts are also recorded in *Task 1.2 – Hypotheses Master List* ("Test results so far").
