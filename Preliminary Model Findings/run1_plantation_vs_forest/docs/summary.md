# Plantation vs natural forest: proof of concept (MapBiomas validation points, 2022)

**Setup.** 5-fold spatial CV (GroupKFold by municipality), out-of-fold predictions, no tuning, seed 42. LR = standardized, balanced. RF = 500 trees, leaf 5, balanced.
**Rows.** The spec run (A) drops every row missing a primary feature. VH is missing for 1,311 of 1,831 rows (72%; Santa Catarina has only 25 rows left), so A keeps **520 rows with only 10 of 40 missed plantations**. I therefore also ran **B = same features without VH, 1,830 rows, 39 missed plantations**. B is the headline below. A is reported for the spec.

## Results, run B (n = 1,830; 428 plantation, 1,402 forest)
"FPR" = real forest predicted as plantation (our dangerous error). FNR = 1 − recall; confusion matrices are in `metrics_all.csv`.

| Model / threshold | AUC (folds) | Acc | Prec | Recall | FPR | Missed plantations caught (of 39) | Real forest flagged (of 1,260) |
|---|---|---|---|---|---|---|---|
| LR, 0.5 | 0.907 ± 0.021 | 0.843 | 0.622 | 0.836 | 0.155 | 21 (54%; CI 39–68) | 176 (14.0%) |
| LR, Youden (0.47) | same | 0.836 | 0.604 | 0.860 | 0.172 | 25 (64%; CI 48–77) | 197 (15.6%) |
| RF, 0.5 | 0.916 ± 0.014 | 0.864 | 0.667 | 0.834 | 0.127 | 18 (46%; CI 32–61) | 133 (10.6%) |
| RF, Youden (0.53) | same | 0.874 | 0.693 | 0.832 | 0.113 | 18 (46%) | 116 (9.2%) |
| RF, FPR ≤ 5% (0.76) *(extra)* | same | 0.879 | 0.802 | 0.643 | 0.049 | 10 (26%; CI 15–41) | 40 (3.2%) |
| **MapBiomas** (binary) | n/a | 0.928 | 0.837 | 0.862 | 0.051 | 0 (0%; CI 0–9) | 0 |

**Run A (spec, n = 520, 94 plantation, 10 missed):** AUC LR 0.929 ± 0.055, RF 0.926 ± 0.045. MapBiomas: acc 0.954, recall 0.840, FPR 0.021. Models at 0.5: recall 0.78–0.82, FPR 0.09–0.13. They catch 4–5 of 10 missed plantations and flag 8–13% of real forest.

## Features (run B; run A for VH)
| Feature | LR odds ratio per 1 SD (95% CI), p | RF permutation ΔAUC | 
|---|---|---|
| sag_oct_nov_mean | 5.67 (4.48–7.18), <0.001 | **0.202 ± 0.015** |
| ever_farm | 2.05 (1.79–2.35), <0.001 | 0.026 ± 0.006 |
| ndvi_aug_oct_mean | 0.94 (0.78–1.12), 0.46 | 0.023 ± 0.007 |
| height_stdDev | 1.82 (1.56–2.14), <0.001 | 0.021 ± 0.015 |
| sag_swing_mean | 0.73 (0.60–0.89), 0.002 | 0.010 ± 0.003 |
| vh_mean (A only) | 0.34 (0.22–0.52), <0.001 | 0.041 ± 0.023 (rank 2 of 6) |

No pair has |r| > 0.8; all VIF < 1.7. **All-candidate run:** AUC +0.029 for both LR (0.907→0.936) and RF (0.916→0.945) in B, better in 5/5 folds. In A it is +0.029 (LR) and +0.037 (RF).

## Takeaways
- **SAG (Oct–Nov) carries most of the signal.** SAG alone gives AUC 0.87 (LR) / 0.84 (RF). The other features add +0.04 / +0.08 (5/5 folds). The window differs from Cropin's Dec–Jan SAG, so compare the two directly.
- **We do not beat MapBiomas overall.** At the same FPR (~5%), RF recall is 0.64 vs 0.86. The only possible value is on the points MapBiomas misses.
- **Missed plantations are only partly separable.** AUC for missed plantation vs real forest is 0.81 (LR) / 0.80 (RF), bootstrap CI ≈ 0.70–0.89. Catching about half costs 9–16% of real forest flagged. That conflicts with the "technically zero" tolerance. At ≤ 3% flagged, only ~26% are caught. Among MapBiomas-forest points flagged by RF at FPR ≤ 5%, 10 are plantation and 40 are real forest.
- **VH helps** on identical rows and folds (ΔAUC LR +0.020, RF +0.035), but it is missing for 72% of rows. Fix the GEE export first. Its effect on missed plantations is untestable (n = 10).
- **Farm-history hypothesis: weak support.** 14/39 missed plantations have ever_farm = 1, vs 188/1,260 real forest (Fisher p = 0.001). It is also 51% for caught plantations, and abandoned farmland can regrow as forest.

## Caveats
Only **39 missed plantations in B (34 municipalities); 10 in A (6 municipalities, 9 in one fold)**. Wilson CIs ignore clustering and are optimistic. Thresholds are picked on the same OOF scores. Municipality blocks do not remove all spatial correlation. Canopy height is from 2020, labels are 2022 only, and the region is southern Brazil only. Edge points were removed, so the edge error category is untested. This sample has 3% missed plantations among MapBiomas-forest points, unlike Cropin's QC set (mostly plantation), so precision and accuracy will not transfer. `ever_farm` comes from MapBiomas C9, so it is not independent of the baseline. Logit p-values are optimistic (spatial autocorrelation). statsmodels was not installable here, so the script's IRLS fallback was used (it matches sklearn to 6e-4). It uses statsmodels automatically if installed.
