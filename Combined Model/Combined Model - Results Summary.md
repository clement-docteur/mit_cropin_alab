# Combined model: results summary (13 Nov 2026)

**Bottom line:** a random forest on our combined features plus AlphaEarth (**B5**) is the first model that clearly beats Cropin's SAG at their alerts. On the QC held-back half, its AUC is 0.93 vs 0.83 for SAG (significant). It clears **62% of plantation alerts vs 53%** for SAG, with 2 forest errors out of 59 vs 3. The best interpretable model (**I2**) did about as well as SAG, but not better.

## What we did

- **Data:** the shared master table (v2). Every teammate idea was rebuilt on our own points and as-of windows: harvest cycles, Meera's winter bands, VH, and land history. Teammates' CSVs were not used; we only kept the ideas they reported as working.
- **Training:** 1,700 MapBiomas points (stable 2013–2022, ≥ 2 interpreters agree, tree filter). Spatially blocked cross-validation by 0.5° block.
- **Model selection:** on the QC development half only (58 forest, 149 plantation), with a rule fixed in advance:
  - forest errors no higher than SAG's;
  - then the highest plantation recall;
  - then fewer forest errors and the simpler model.
- **Final test:** three finalists scored **once** on the QC held-back half (59 forest, 148 plantation). The threshold was set for 5% forest error on MapBiomas points.

## Models compared (development half)

12 models in total:

- **Interpretable:** logistic regression (full and sparse), a spline model (works like a GAM), and a decision tree.
- **Black box:** random forest, gradient boosting, and a neural network, each with and without AlphaEarth; plus logistic regression on AlphaEarth alone.

## Final test (held-back half)

| Model | AUC [95% CI] | Forest errors | Plantation recall |
|---|---|---|---|
| SAG (reference) | 0.83 [0.75–0.89] | 3/59 | 53% |
| I2: sparse logistic regression, 14 features | 0.87 [0.82–0.92] | 2/59 | 46% |
| **B5: random forest, 32 features + AlphaEarth** | **0.93 [0.89–0.96]** | **2/59** | **62%** |

**I2's 14 features:**

- SAG Sep–Nov and SAG seasonal timing;
- canopy height (ETH);
- radar VH−VV;
- Landsat history: largest NBR drop, CCDC breaks, and years of continuous forest;
- Hansen: loss within 150 m and tree cover 2000;
- NBR variability (harvest-cycle idea);
- Natural Forests of the World;
- MapBiomas: the 10 m map, years of continuous natural cover, and number of land-cover changes.

## What we learned

1. **AlphaEarth plus our features beats either one alone at the alerts.** AlphaEarth alone scored AUC 0.80 in Phase 7, and our features without AlphaEarth scored 0.87 on the development half.
2. **At the strictest forest tolerance, simple models held up better** (development half). With forest errors held at 3 of 58, logistic regression cleared 44% of plantation alerts, SAG 34%, and random forest + AlphaEarth 26%.
3. **A threshold set on MapBiomas points doesn't transfer reliably.** Forest errors at the alerts ranged from 3% (held-back half) to 17% (development half). Weighting the threshold toward alert-like points barely changed this.
4. **MapBiomas-derived features help forest safety at the alerts but cost recall.** Without them, B5 cleared 68% of plantation alerts but made 5 forest errors instead of 2. On MapBiomas points these features are circular, so their 0.99 scores there mean little.
5. **Ideas that didn't work in our rebuild:** using "forest every year since 1985" as a veto (more forest errors, not fewer), and the MapBiomas 10 m map as a feature in logistic regression.

## Follow-up: Cropin's SAG + a few features (development half only)

Run after the held-back test, so the QC numbers below are development-half only and not confirmed. Rule fixed before running: start from Cropin's SAG (one season, Oct–Jan) and add one feature at a time, choosing the one that most raises cross-validated AUC on MapBiomas training points; stop after 5 additions or when the gain is < 0.005. QC alerts were used only for reporting.

| Step | Added | MapBiomas AUC | QC AUC | QC forest errors | QC recall |
|---|---|---|---|---|---|
| 0 | Cropin's SAG alone | 0.84 | 0.79 | 10/58 | 63% |
| 1 | + NBR variability (harvest-cycle idea) | 0.97 | 0.82 | 14/58 | 66% |
| 2 | + Natural Forests of the World | 0.98 | 0.83 | 10/58 | 62% |
| 3 | + Hansen forest loss within 150 m | 0.99 | 0.83 | 11/58 | 67% |

- The same features were picked in every cross-validation fold (nested MapBiomas AUC 0.985). No MapBiomas-derived feature was picked.
- **Big gain on MapBiomas points, small gain at the alerts, and no fewer forest errors.** Selection on MapBiomas points favours history features, which transfer poorly to Cropin's alerts.
- At the alerts, SAG + 3 (AUC 0.83) sits below I2 (0.85), I1 (0.87), B5 (0.87), and Phase 7's Lean 4 (0.85), which builds on SAG with seasonal spectral features instead of history.
- Building on SAG properly would mean choosing additions on alerts, which needs more labelled alerts from Cropin.

## Caveats

- The held-back half was already seen during teammates' feature screening, so expect somewhat lower numbers on fresh alerts. **Cropin's blind holdout is the clean test.**
- With about 60 forest points per half, 2 vs 3 forest errors is not a proven difference.
- Cropin has no labels for forest it misses, so none of this shows the overall forest error.
- B5 needs AlphaEarth plus 32 features per pixel, so it costs more to map than SAG. That matters for the map view Cropin asked for.

## Next steps

1. Map view on 3–5 pilot tiles: B5, I2, SAG, and MapBiomas 10 m side by side.
2. Ask Cropin to score B5 and I2 on their blind holdout.
3. Choose how to set the threshold at the alerts. This needs more labelled alerts, especially forest.

Files: `Combined Model/` (protocol, scripts, results).
