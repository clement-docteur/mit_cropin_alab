# Plantation vs natural forest: proof of concept (southern Brazil)

Compares LR and RF on MapBiomas validation points (2022) to separate plantation from
natural forest. Spatial 5-fold CV (GroupKFold by municipality). See `docs/summary.md`.

## Layout
- `scripts/poc_plantation_vs_forest.py`: full analysis (tasks 1-10 + extra diagnostics)
- `Models/`: fitted models (`joblib`) + `*_meta.json` (features, thresholds, versions)
- `results/`: metrics, out-of-fold predictions, feature tables, run log
- `figures/`: ROC, PR, permutation importance, correlation plots
- `docs/summary.md`: one-page summary

## Runs
- `A_withVH_spec`: spec run, 6 primary features incl. Sentinel-1 VH, 520 rows (VH missing for 72% of rows)
- `A_noVH_ablation`: same 520 rows, without VH
- `B_noVH_allrows`: without VH, 1,830 rows (headline run)

## Using a saved model
```python
import joblib, json, pandas as pd
m = joblib.load("Models/B_noVH_allrows_RF_primary.joblib")
meta = json.load(open("Models/B_noVH_allrows_meta.json"))
X = df[meta["features_primary"]]            # ever_farm = (ever_farm_mean >= 0.5).astype(int)
p = m.predict_proba(X)[:, 1]                # probability of plantation
flag = p >= meta["thresholds_from_oof"]["RF"]["FPR<=5%"]
```
## Reproduce
`python scripts/poc_plantation_vs_forest.py --data <features.csv> --out <dir>`
(writes all outputs, including `Models/`, under `<dir>`; the input CSV is not committed here).

## Important
- Models are fit on ALL rows. Judge performance from the out-of-fold results in `results/`, never from these fits.
- Thresholds were chosen on the same out-of-fold scores, so they are optimistic.
- Not validated on Cropin QC points yet. Do not use for production decisions.
