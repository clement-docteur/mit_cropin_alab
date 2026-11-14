# Combined model search – protocol v1 (locked 13 Nov 2026, before any model was fit)

**Data:** `master_features_v2.csv` with all `hc_` columns (teammates' data) replaced by `rh_` columns rebuilt on our data (`rebuild_harvest.py`). `r2_` columns (teammates' data) are not used; their ideas are covered by our own columns (VH → `nc_F_VH_median`; height → `nc_E_eth_h`) or by pending exports (farm history → batch M2). AlphaEarth = our as-of export (`master_alphaearth_v2.csv`).

**Pending:** `mm_B6_win` (batch M1) and land history (batch M2). This run is **preliminary**; the search is re-run once both arrive, and finalists are chosen only then.

**Feature pools**
- **P_int (30 features):** every `feature`-role column of v2 with `hc_` → `rh_` (n_cycles, s_std, s_frac_below_forest, abrupt_start; `rh_max_drop` only if |ρ| < 0.85 with n_cycles), `hc_tc2000` → our `nc_hansen_tc2000`; `hc_hansen_gain` dropped (not in our data); `mm_B6_win` excluded until filled.
- **P_bb:** P_int + 64 AlphaEarth dimensions.
- `rh_forest_since_1985` is a **veto rule** (can only turn a plantation call into forest), tested on top of finalists, never as an input.

**Models** (fit on the 1,700 `train_eligible` points; label 1 = plantation)
| ID | Type | Pool | Settings (small grids chosen by inner grouped CV on training folds, by AUC) |
|---|---|---|---|
| REF | SAG reference | `nc_A_SAG` | Cropin's index alone, score = SAG |
| I1 | Logistic regression L2 | P_int | C ∈ {0.1, 1} |
| I2 | Logistic regression L1 (sparse) | P_int | C ∈ {0.02, 0.05, 0.1, 0.3} |
| I3 | Spline GAM (additive splines + L2 logistic) on I2's selected features | P_int | 4 knots, cubic, C ∈ {0.1, 1} |
| I4 | Decision tree | P_int | depth ∈ {3, 4}, min leaf 20 |
| B1 | Random forest | P_int | 500 trees, min leaf ∈ {5, 20} |
| B2 | Gradient boosting (HistGB) | P_int | depth ∈ {2, 3}, learning rate 0.05, 300 iterations, early stopping |
| B3 | Neural network (MLP 32-16) | P_int | alpha ∈ {1e-3, 1e-2}, early stopping |
| B4 | Logistic regression L2 | AlphaEarth only | C ∈ {0.1, 1} |
| B5–B7 | RF / HistGB / MLP | P_bb | as B1–B3 |

**Evaluation (same as Phase 7)**
- MapBiomas: 5-fold GroupKFold by `block` (nested for tuning); out-of-fold AUC with block bootstrap; recall at 5% forest error.
- Threshold: set on MapBiomas out-of-fold scores at 5% forest error, then applied to QC dev (model refit on all 1,700).
- QC dev (58 forest, 149 plantation): AUC with 0.25° block bootstrap; **forest errors (count / 58, Wilson CI) and plantation recall, always together.**
- Edge category: MapBiomas OOF forest error for points ≤ 10 m from non-tree land.
- Models using MapBiomas-derived features are also reported without them.

**Selection rule (business case: forest protection first, then plantation alarms cleared)**
1. Constraint: QC-dev forest errors ≤ the SAG reference's count at its own threshold.
2. Primary: QC-dev plantation recall at the MapBiomas-set threshold.
3. If two models are within 5 percentage points of recall (≈ 7 plantations), prefer the one with fewer QC-dev forest errors, then the simpler one.
4. Finalists: best interpretable (I*), best black box (B*), and REF. Scored once on QC held-back after the final search (held-back reported as "seen during feature screening").

**Known limits stated in advance:** QC dev has 58 forest points (one error ≈ 1.7 pp); 11 models compared on it, so the winner's QC-dev numbers are optimistic. Cropin has no labels for forest it misses, so no result here shows the overall forest error.

---

## Addendum v2 (locked 13 Nov 2026, after the preliminary run, before the final run)

**Changes, approved by Nathan:**
1. **Pool additions:** `mm_B6_win` (batch M1) and three land-history features from batch M2, deduplicated with the same rule (greedy by MapBiomas AUC, drop |ρ| ≥ 0.85): `lh_natural_run_years` (kept), `lh_n_changes` (kept), `lh_ever_open` (kept); `lh_forest_all_years` (ρ 0.97 with natural run), `lh_ever_farm` and `lh_farm_years` (ρ ≥ 0.87 with ever open) dropped. All `lh_` features are MapBiomas-derived.
2. **Alert-weighted threshold reported alongside** the standard one: 5% forest error on MapBiomas out-of-fold scores, with each training point weighted by how much its context (patch size, distance to non-tree land, tree fraction at 50/100/250 m, slope) looks like a QC dev point of the same class (Phase 7 method). **Finalist selection still uses the standard threshold** (rule unchanged).
3. **Without-MapBiomas variant** for each finalist drops `nc_H_mapbiomas10m_plantation` and all `lh_` features.
4. The veto rule (`rh_forest_since_1985`) is not re-tested: it failed in the preliminary run (more forest errors at alerts, lower recall).

Everything else (models, grids, folds, metrics, selection rule, finalists, single held-back scoring) is unchanged. The preliminary results were seen before this addendum, which adds selection optimism; stated here so readers can weigh it.

---

## Finalists (recorded 13 Nov 2026 from `search_v2_results.csv`, before the held-back half was scored)

Rule applied: constraint QC-dev forest errors ≤ REF's 10/58 (all pass except B4, 11/58).
- **Interpretable:** I2 has the best recall (56.4%); I1 and I3 are within 5 points; I2 and I3 tie on forest errors (8/58); I2 is simpler (linear vs splines on the same 14 features) → **I2 (sparse L1 logistic regression, 14 features).**
- **Black box:** B5 has the best recall (63.1%); B7 is within 5 points with more forest errors (9 vs 8) → **B5 (random forest, 32 features + 64 AlphaEarth).**
- **Reference:** SAG.

Held-back scoring (once): the three finalists, fit on all 1,700 training points, thresholds from MapBiomas out-of-fold scores (standard, and alert-weighted alongside). Secondary, labelled: I2 and B5 without MapBiomas-derived features. Comparisons vs SAG: paired block-bootstrap AUC difference (0.25° blocks), forest errors with exact McNemar on forest points.
