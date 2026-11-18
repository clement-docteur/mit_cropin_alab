# Feature Exploration – Summary of Findings (Phases 0–7, final)

*Prepared by Nathan Chu (with Claude) · 11 November 2026 (updated after the final test) · Analyses are our own (Google Earth Engine + Python) unless a source is given · **[inference]** = our interpretation, not tested · Detailed evidence: `Wave 1 Report`, `Wave 2 Report`, `Wave 3 Report`, `Phase 7 Protocol`, `Run Log.md`, and `results/` in the `Feature Exploration` folder · Method: `00 Workflow Plan.md`*

---

## 1. Summary

1. **On MapBiomas-like data, four simple features clearly beat the baseline, and this held up in the final sealed test.** The features are Cropin's SAG colour signal measured in spring, the timing of the yearly colour cycle, summer canopy moisture, and late-spring greenness. On the sealed MapBiomas lockbox, AUC rose from 0.84 to 0.92 (significant), and plantation recall from 36% to 67% at about 5% forest error.
2. **At Cropin's alerts, the gain is small and not proven.** On the sealed half of Cropin's QC points (59 forest, 148 plantation), AUC rose only from 0.81 to 0.83 (not significant), with 3 vs 2 forest errors. The larger gain we saw on QC development points (0.78 → 0.85) did not replicate. It was partly small-sample noise and partly optimism from choices that used those points.
3. **Public maps and AlphaEarth are not safe at Cropin's alerts.** They look strong on MapBiomas points but, at the alerts, either miss most plantations (MapBiomas map 17%, JRC forest types 7%) or call too much forest "plantation" (AlphaEarth 15%; NFW 14–39% depending on threshold). **No NFW threshold is safe at alerts in our region.**
4. **Cropin's SAG works better in spring.** SAG measured over September–November separates plantations from natural forest better than the December–January window Cropin uses on MapBiomas points (AUC 0.88 vs 0.83), and is equal at Cropin's points.
5. **Land-history features are dangerous on their own.** They call 47% of disturbed-and-regrown natural forest "plantation", which is the case Cropin warned about. Forest edges and younger forest are the other weak spots for every model.
6. **Of the 43 numbered hypotheses:** 10 are supported (most only on MapBiomas points), 11 are not supported, 7 are weak, 1 is context-dependent, 9 could not be tested with our data and access, 4 are not yet tested, and 1 (H17) is descriptive.
7. **The main open question is what makes Cropin's alerts different.** Plantations at Cropin's alerts look more like natural forest than MapBiomas plantations do. Answering this needs more labelled alerts, especially forest ones, from Cropin.

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
| MapBiomas **lockbox** | 20% of 0.5° blocks, chosen at random by state | 436 points (427 after the tree filter: 332 forest, 95 plantation) | Sealed until the final test; used once in Phase 7 |
| Cropin QC points | Small areas at recent clearings, labelled forest or plantation by Cropin from historical Google Earth imagery | 415 after removing near-duplicates; dev half 207 (58 forest, 149 plantation) | Checking that results carry over to real alerts. **Never used for training** |
| QC **held-back half** | The other half, split by 0.25° blocks | 208; 207 usable (59 forest, 148 plantation), since one point was cleared in 2022 with no earlier Sentinel-2 season | Sealed until the final test; used once in Phase 7 |

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

## 6. Models on development data

Development data. The final, independent test is in §8. Its Cropin-point results are weaker than these.

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

### 7.3 Error breakdown (Phase 6, development data)

| Group (MapBiomas dev) | n forest | Baseline | Lean 4 | P4 |
|---|---|---|---|---|
| Within 10 m of non-tree land (edges) | 41 | 14.6% | **22.0%** | 17.1% |
| More than 100 m from non-tree land | 682 | 3.4% | 3.7% | 4.0% |
| Younger forest (forest only since 1986–2012; HC2) | 123 | 8.1% | 8.9% | 7.3% |
| Only 2 of 3 interpreters agree | 281 | 9.6% | 9.6% | 11.0% |
| Santa Catarina | 432 | 6.7% | 6.0% | 7.4% |

- **Error clusters** sit in the highlands (Serra Catarinense, RS highlands, SC midwest) and on the coast (Paraná coast, SC north coast). **[inference]** These are araucaria/pine country and dense coastal rainforest.
- **Where the MapBiomas map misses Cropin's plantations (Task 1.5):** in areas where the 10 m map shows no plantation class, it catches 0 of 30 QC plantations, against 24% elsewhere (M3).

## 8. Final test (Phase 7)

The model, thresholds, test sets, and comparisons were locked in `Phase 7 Protocol` before the sealed data were opened, and the data were opened once. Lean 4 was the declared main model and P4 the declared challenger. An independent reviewer reproduced every number.

| Model | QC held-back AUC [95% CI] | QC forest error / recall | Lockbox AUC [95% CI] | Lockbox forest error / recall |
|---|---|---|---|---|
| Baseline | 0.812 [0.75–0.87] | 2/59 (3.4%) / 46% | 0.843 [0.79–0.89] | 16/332 (4.8%) / 36% |
| **Lean 4 (main)** | 0.829 [0.77–0.88] | 3/59 (5.1%) / 50% | **0.917** [0.87–0.96] | 20/332 (6.0%) / **67%** |
| P4 (challenger) | 0.849 [0.78–0.90] | 2/59 (3.4%) / 53% | 0.938 | 18/332 (5.4%) / 71% |
| Baseline + NFW | 0.801 | 6/59 / 40% | 0.965 | 18/332 / 84% |
| AlphaEarth + logistic (benchmark) | 0.800 | 9/59 (15.3%) / 55% | 0.936 | 23/332 / 76% |
| NFW alone, threshold 0.52 | 0.739 | 12/59 (20.3%) / 57% | 0.975 | 46/332 / 94% |
| MapBiomas 10 m map | 0.584 | 0/59 / 17% | 0.891 | 20/332 / 84% |
| JRC forest types 2020 | 0.537 | 0/59 / 7% | 0.874 | 7/332 / 77% |

The QC held-back half has 59 forest and 148 plantation points; the lockbox has 332 forest and 95 plantation points after the tree filter. Forest error and recall are at each model's locked 5% threshold.

**Pre-declared comparisons:**

| Comparison | Result |
|---|---|
| Lean 4 vs baseline, QC AUC | +0.017 [−0.04, +0.07], p = 0.31: **not supported** |
| Lean 4 vs baseline, QC forest errors | 3 vs 2: **not met** |
| P4 vs baseline, QC | +0.037 AUC, p = 0.07 (not significant); forest errors 2 vs 2 |
| Lean 4 / P4 vs baseline, lockbox AUC | +0.074 / +0.095, both p < 0.001: **supported** |
| Lean 4 / P4 vs baseline, lockbox forest errors | 20 / 18 vs 16 (not significantly different) |

**Reading:**

- The features capture something real about plantations, but mostly the kind of plantation in MapBiomas's sample.
- At Cropin's alerts, plantations look more like natural forest. In the sealed QC set, their SAG-phase values sit between MapBiomas forest and MapBiomas plantation, and they get NFW scores of about 109 instead of 0.
- The earlier finding that "thresholds do not transfer" did not repeat on the sealed half: the baseline's QC forest error was 17% on dev but 3.4% on the sealed half. **[inference]** That finding was largely small-sample noise.
- P4 did better than Lean 4 on both sealed sets, but it was declared the challenger, so this is reported as such, not swapped in.

## 9. Limits

- **Small, biased QC sample.** 58 forest points in the dev half and 59 in the sealed half. A forest error of 3/59 has a 95% CI of about 2–14%, and an AUC difference of 0.02 cannot be detected reliably at this size. QC forest is only forest where Cropin already detected a clearing, and **Cropin has no labels for forest it misses**.
- **QC dev was not independent.** It informed several choices (flags, alert-like weights, model comparison), which helps explain why dev-QC results were better than the sealed-half results. Both sealed sets have now been used once and are no longer untouched.
- **Labels.** MapBiomas labels come from 30 m Landsat interpretation. Points where only 1 of 3 interpreters agreed were left out of training and reported separately. QC labels have not been visually re-checked by us.
- **Different years.** MapBiomas features come from 2020–22 (dry: 75–97% of normal rainfall) and QC features from 2023–25 (wet: 100–114%). Some differences between the sets may be weather.
- **Region.** All findings are for southern Brazil. SAG timing and phase in particular depend on local phenology, and should be retested before a global layer.

## 10. Next steps

| Step | Why |
|---|---|
| **Ask Cropin for more labelled alerts,** especially forest ones, and for them to score our layer on their blind holdout | The only way to settle whether the features help at alerts. With about 60 forest points, small gains can't be detected. |
| Ask Cropin for the QC clearing dates | We estimated them from greenness; exact dates make the before-clearing features cleaner |
| Exploratory, clearly labelled as post-hoc: why plantations at alerts look like forest | Ideas for the next model (for example, younger or recently replanted stands, mixed pixels in small patches) |
| Visual check of the forest points in error (two labellers) | Tells us whether some errors are label errors |
| For Cropin now | Test a Sep–Nov SAG window alongside Dec–Jan, a cheap change to an existing method. Its benefit at alerts is not proven. |

## 11. Files

All in `Cropin - A Lab/Feature Exploration/`:

- `00 Workflow Plan` (method), `Phase 7 Protocol` (locked final test), `Run Log.md` (every run, decision, and review)
- `Wave 1 Report`, `Wave 2 Report`, `Wave 3 Report` (detailed results)
- `results/` (tables, including `results/phase6` and `results/phase7`), `scripts/` (all code, including Google Earth Engine scripts)
- Hypothesis verdicts are also recorded in *Task 1.2 – Hypotheses Master List* ("Test results so far").
