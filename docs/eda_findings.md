# QC error analysis: findings so far (Meera, 2026-10-29)

**Data.** Cropin's 433 hand-checked alerts: 310 plantation harvests (false alarms) and 123 real
deforestation. Every case was already called forest by Cropin's layer and already cleared, so the
sample is **biased and pre-filtered**. 13 rows repeat another polygon exactly; results barely
change without them. Everything below is **exploratory**: features were chosen on these same
cases. Full working: `notebooks/01_qc_error_analysis.ipynb` and `02_cropin_layers_and_SAG.ipynb`.

**How to read the numbers.**
- **AUC**: the chance a random plantation scores more plantation-like than a random forest (0.5 = coin flip).
- **[a–b]**: bootstrap 95% CI.
- **"Removed at ≤ 5%"**: the share of plantation false alarms a rule would remove while hiding at
  most 5% of real deforestation. It is measured with a threshold chosen inside each training fold
  (nested) and spatially blocked cross-validation, as mean ± SD over 20 seeds.

## What MapBiomas says about these cases
| Product, year | Plantations called natural forest | Plantations caught |
|---|---|---|
| C9 30 m, 2020 (EUDR cutoff) | **91%** (282/310) | 3.5% (11) |
| C10 30 m, 2020 | 100% | 0% |
| 10 m C2 / S2-beta, 2020 | 82% / 79% | 11% / 17% |
| 10 m C2 2023 / S2-beta 2022 | 69% / 68% | 25% / 29% |
| Julia: 10 m C4, 2025 (n = 419) | 82% | 18% |

All are valid for their own question. MapBiomas relabels more plantations in later years. Because
the alerts were selected *through* MapBiomas-based layers, the fair question is what new
information adds on top.

## Single signals (plantation vs forest)
| # | Signal | Result | After Holm correction |
|---|---|---|---|
| 1 | Google Natural Forests prob. | lower in plantations, AUC 0.72 [0.66–0.77]; < 0.10 removes 16% but hides 5% of forest | significant |
| 2 | ETH canopy height 2020 | taller plantations (24.2 vs 21.2 m), AUC 0.71 [0.65–0.76] | significant |
| 3 | Height variability | plantations *more* uneven: 30 m circle AUC 0.66, **and also inside the polygon** with Meta 1 m (AUC 0.64). Not just a surroundings artefact (caveat: Meta's mixed image dates) | significant |
| 4 | MapBiomas land history | missed plantations are less often forest every year (50% vs 69%, OR 0.45) and more often ex-farmland (46% vs 30%, OR 2.0) | significant |
| 5 | Patch size | no difference (AUC 0.50) | n.s. |
| 6 | Yearly NDVI | no dips difference (AUC 0.51); tiny greenness difference | dips n.s. |
| new | **SAG, Oct–Jan mean** (Cropin) | **AUC 0.87 [0.83–0.91]**; winter (Jun–Aug) 0.78; robust to dropping the 2025/26 season | significant |

## Combined model (logistic regression, standardized features)
| Model | Spatial CV AUC | Removed at ≤ 5% forest hidden |
|---|---|---|
| MapBiomas C9 2020 class 9 | – | 3.5% (hides 0%) |
| Notebook 01: NF + height + height SD + class changes + years forest | 0.81 ± 0.01 | **30% ± 2** |
| SAG Oct–Jan alone | 0.86 | 34% ± 2 |
| **Notebook 01 features + SAG** | **0.89** | **49% ± 3** |
| Cropin's SAG posterior, fixed ≥ 0.8 rule | 0.87 (ranking) | catches 68% but **hides 16%** of forest |

**Corrections to the midterm numbers.**
- **36.5% → about 30%.** The old figure reproduces exactly, but it came from one lucky split (3rd of
  20, range 26–37%) and from picking the threshold on the scored predictions.
- **AUC 0.82 holds**, and is 0.81 under spatial blocking.

## SAG specifics
- **Fitted curves.** Fitting Cropin's bell curves to our labels gives forest ≈ 153° and plantation
  ≈ 161° (vs 150/160 hard-coded), with SD ≈ 6 (vs 9). It does **not** fix the threshold problem:
  55% caught, 13% of forest flagged.
- **The 0.9 clamp.** It creates ties at the top (134 polygons at exactly 0.9, including 6 forests),
  so the posterior can't be run at a ≤ 5% operating point. Raw mean SAG works better.
- **Dark native pine.** The 20 forests SAG wrongly flags lean towards "forest every year", but
  land history doesn't significantly separate them.

## Caveats
- **Biased, small sample.** Precision depends on the unknown real class mix, and "5% of forest" is
  6 polygons.
- **Measurement limits.** MapBiomas modes are 30 m on sub-pixel polygons. ETH is from 2020 and the
  clearings from 2026. Meta heights are low (median 8 m) and have mixed dates.
- **The QC alerts only partly match Cropin's shared layer.** Only 55% of plantations pass its final
  flag at polygon level, and 28% are MapBiomas 10 m plantation in 2020.

## Asks and next steps
- **Ask Cropin:**
  - which forest-layer version and year produced the alerts;
  - why 12 polygons are repeated;
  - the CCDC script;
  - a week of fresh alerts.
- **Ask Julia:** confirm 10 m C4 2025 and the 14 dropped cases, and build one shared feature table.
- **Next analyses:**
  - validate SAG + features on the 2,290 MapBiomas reference points;
  - map a small region of interest and inspect it, especially for dark native pine;
  - try CCDC / harvest-cycle features.
