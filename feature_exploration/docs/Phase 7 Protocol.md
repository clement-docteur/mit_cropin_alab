# Feature Exploration – Phase 7 Protocol (final test, locked)

*Prepared by Nathan Chu (with Claude) · Locked 11 November 2026, before any sealed data were opened · Approved by Nathan Chu · Plan: `00 Workflow Plan.md` (v4.1) · Decisions: `Run Log.md`*

---

This protocol fixes the model, the thresholds, the test sets, the benchmarks, and the metrics **before** the sealed data are opened. Nothing in it changes after opening. Every result is reported as it comes out, including results that look bad.

## 1. Models

| Role | Model | Features | Training |
|---|---|---|---|
| **Main** | **Lean 4** | SAG phase (sin), SAG Sep–Nov, summer NDMI, Oct–Jan NDVI | Logistic regression (L2, C = 1, median imputation, standardized), trained once on all MapBiomas dev points with interpreter agreement ≥ 2 and summer NDVI > 0.6 (1,700 points) |
| Challenger | P4 | Lean 4 plus NDRE phase (sin), SWIR phase (sin, cos), SAG 3-year mean, SAG Dec–Jan, SAG Oct–Jan composite | Same |

No refitting, feature change, or recalibration after opening. **No edge guard** (decision 11 Nov 2026); edges are reported as their own category.

## 2. Thresholds (fixed now)

Set on MapBiomas dev blocked out-of-fold scores. The values and the fitted coefficients are stored in `results/phase7/p7_locked_thresholds.json` (SHA-256 `42d032e3b6ea62e36eb6054690e9c67da3d71d3203fd900161745095ea02c36c`).

| Model | 1% forest error | 2% | **5% (primary)** | 5%, alert-like weighted |
|---|---|---|---|---|
| Lean 4 | 0.6917 | 0.5998 | **0.4215** | 0.4622 |
| P4 | 0.7510 | 0.5879 | **0.4057** | 0.4376 |
| Baseline | 0.7206 | 0.6227 | 0.4808 | 0.4989 |
| Baseline + NFW | 0.7723 | 0.6269 | 0.4340 | 0.4692 |

## 3. Test sets (sealed until now)

| Set | Size | Filters |
|---|---|---|
| Cropin QC held-back half | 208 points (59 forest, 149 plantation) | Same as QC dev: features from before the estimated clearing month |
| MapBiomas lockbox | 436 points (334 forest, 102 plantation), 44 blocks | Same as dev: stable class 2013–2022, summer NDVI > 0.6. Points with agreement = 1 are reported separately |

## 4. Benchmarks

| Benchmark | Note |
|---|---|
| Baseline: SAG + Meta height + height spread | Decision-rule baseline |
| Baseline + NFW | Comparison |
| NFW alone at thresholds 0.30, 0.50, 0.52, and 0.70 (B0 ≥ 75, 125, 130, 175) | Also answers Cropin's NFW threshold question for our region (MapBiomas-derived: unknown) |
| MapBiomas 10 m map (plantation class, cutoff year) | MapBiomas-derived: yes; judged on QC |
| JRC global forest types 2020 ("planted") | Uses NFW as input; judged on QC |
| AlphaEarth embeddings + logistic regression (H28) | **Black-box benchmark only.** Trained on MapBiomas dev embeddings (as-of year), same protocol |

## 5. Metrics

- AUC with 95% block-bootstrap CI (QC: 0.25° blocks; lockbox: 0.5° blocks).
- At each threshold: forest error (real forest called plantation) with Wilson 95% CI, and plantation recall, always reported together.
- The same groups as Phase 6: distance to non-tree land, MapBiomas border flag, interpreter agreement, state, younger forest (HC2 proxy), and disturbed natural forest.

## 6. Pre-declared comparisons

1. Lean 4 QC AUC > baseline QC AUC (paired block bootstrap, one-sided, α = 0.05).
2. Lean 4 QC forest error at the primary threshold ≤ the baseline's (reported with both counts and an exact McNemar test on the forest points).
3. The same two comparisons for P4.
4. The same comparisons on the MapBiomas lockbox.

## 7. Known limits, stated in advance

- The QC held-back half has 59 forest points, so forest-error CIs will be roughly ±5–10 percentage points.
- QC forest is only forest where Cropin already detected a clearing. **Cropin has no labels for forest it misses**, so no result here shows the overall forest error. Cropin's blind holdout would be the stronger test.
- QC dev informed several earlier choices. The held-back half and the lockbox did not.
