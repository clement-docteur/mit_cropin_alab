# Run 2: updated radar (VH) + AlphaEarth embeddings

**Data.** Merge on `pt_id`: 1,831 of 1,831 matched, no duplicates or unmatched rows; labels and `CD_MUN` agree between files. Missing values: only NDVI/NDMI in 1 row (0.05%); that row is a missed plantation and is dropped, so **n = 1,830 for every model** (39 missed plantations in 34 municipalities). **VH/VV are now 0% missing (was 71.6%).** VIF: `vv_mean` dropped (20.4; `vh_mean` 19.8); kept set A = 6 features, all VIF ≤ 1.62.
**Coding.** 1 = forest everywhere (training label, model score = P(forest), logit outcome, output files). **Rule.** Forest = positive; FP = plantation predicted as forest. Threshold chosen on inner out-of-fold scores in each training fold so forest recall = MapBiomas's. MapBiomas: forest recall 0.899 (1,260/1,402), **FP = 39 (9.1% of 428 plantations)**. CIs: bootstrap over municipalities (they do not include threshold-selection noise).

| Model (features) | Forest recall | FP (rate) | FP reduction vs MapBiomas: count (%) [95% CI of %] | Missed plantations rejected (of 39) | New FP on other plantations | AUC |
|---|---|---|---|---|---|---|
| RF, A (handcrafted) | 0.898 | 81 (18.9%) | −42 (−108%) [−208, −52] | 16 | 58 | 0.924 |
| RF, B (embeddings) | 0.900 | 45 (10.5%) | −6 (−15%) [−69, +18] | 21 | 27 | 0.959 |
| RF, C (A + emb.) | 0.904 | 33 (7.7%) | +6 (+15%) [−24, +41] | 24 | 18 | 0.964 |
| L2-LR, B | 0.905 | 30 (7.0%) | +9 (+23%) [−14, +49] | 25 | 16 | 0.968 |
| L2-LR, C | 0.899 | 27 (6.3%) | +12 (+31%) [−3, +54] | 25 | 13 | 0.975 |
| ref: plain LR, A | 0.902 | 93 (21.7%) | −54 (−138%) [−254, −73] | 16 | 70 | 0.914 |
| ref: RF, A0 (no radar) | 0.895 | 82 (19.2%) | −43 (−110%) [−204, −54] | 17 | 60 | 0.915 |

**Cluster-robust logit (plain LR on A, outcome = FOREST, i.e. 1 = forest; OR > 1 = more forest-like; OR per 1 SD):** `vh_mean` 1.99 (robust CI 1.61–2.46), `sag_oct_nov_mean` 0.19 (0.15–0.26), `ever_farm` 0.45 (0.39–0.53), `height_stdDev` 0.56 (0.46–0.69), all p < 0.001; `sag_swing_mean` 1.37 (1.10–1.72), p = 0.006; `ndvi_aug_oct_mean` 1.06 (0.84–1.33), p = 0.62. Robust SEs are 6–28% larger than Wald. IRLS converged in 8 iterations (last change 3e-16), matches sklearn to 2e-8, max SE 0.12 (no separation).

## Answers
1. **VH: no reliable FP gain.** RF with vs without VH: FP 81 vs 82 (−1; CI −10 to +8). The RF's own noise is ±3 FP (identical features, columns reordered: 78 vs 81). Plain LR: −13 FP (CI −26 to −1), borderline. AUC +0.009 (RF). The earlier +0.035 came from the 520-row subset.
2. **Embeddings beat handcrafted:** RF B vs A removes 36 more FP (CI −54 to −19) at equal recall.
3. **C vs B:** RF yes, −12 FP (45→33; CI −20 to −4). L2-LR no, −3 (CI −9 to +3).
4. **Significant with robust SEs:** all except `ndvi_aug_oct_mean`.
5. **With embeddings in the model (RF, C):** embeddings AUC drop 0.272 ± 0.016 (FP at matched recall 34→350). `sag_oct_nov_mean` keeps a small unique role (0.012 ± 0.003; FP 34→50) vs 0.177 ± 0.024 in A. `vh_mean` loses it (0.000 ± 0.0004 vs 0.015 ± 0.008 in A).

## Caveats
- **No model clearly reduces FP vs MapBiomas** (best: −31%, CI −3% to +54%).
- Only **39 missed plantations**, one year (2022), one region, radar 2021–2022, canopy height 2020.
- Matched recall 0.899 accepts MapBiomas's ~10% forest loss, far from "technically zero".
- **By state (exploratory, not recall-matched):** FP MapBiomas vs embedding models: PR 10 vs 13–20 (worse), SC 12 vs 8–15, RS 17 vs 4–10 (better, but RS recall is lower: 0.91–0.93 vs 0.948).
- Embeddings include spatial context, so neighbouring municipalities can share information.
