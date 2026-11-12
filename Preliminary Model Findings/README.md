# Plantation vs natural forest: proof of concept (southern Brazil: PR, SC, RS)

Can remote-sensing features separate plantation from natural forest better than MapBiomas?
Labels = MapBiomas validation points (2022, n = 1,830 after dropping one row). Spatial 5-fold
cross-validation, grouped by municipality (`CD_MUN`).

## Layout
| Folder | Contents |
|---|---|
| `run1_plantation_vs_forest/` | First analysis (6 handcrafted features; Sentinel-1 VH was 72% missing). Scripts, saved models (`Models/`), results, figures, summary. Has its own README. |
| `run2_vh_embeddings/` | Current analysis: VH fixed (2021-2022 radar), plus AlphaEarth embeddings. `scripts/`, `results/`, `figures/`, `docs/summary_run2.md`. |

## Conventions (read this first)
- **Run 1 codes plantation = 1. Run 2 codes forest = 1** everywhere (training label, model score = P(forest), logit outcome, out-of-fold files).
- Run 2 **FP = a plantation predicted as forest** (Cropin's false positive: a harvest flagged as deforestation).
- Thresholds are chosen inside each training fold so a model's **forest recall matches MapBiomas's** (about 0.90).
- Run 2 CSVs and `summary_run2.md` say **"FP reduction"** (positive = fewer FP than MapBiomas). The slides use **"Change in FP"** (negative = fewer FP). Same numbers, opposite sign.
- Odds ratios and CIs are cluster-robust by municipality. For an odds ratio, "no effect" is **1.0**; for a difference, it is **0**.

## Run 2 results (MapBiomas: forest recall 0.899, 39 FP)
A = handcrafted features (6) · B = AlphaEarth embeddings (64) · C = A + B

| Model | Forest recall | AUC | Accuracy | FP | Change in FP vs MapBiomas (95% CI) |
|---|---|---|---|---|---|
| RF A | 0.898 | 0.924 | 0.878 | 81 | +108% (+52 to +208) |
| RF B | 0.900 | 0.959 | 0.899 | 45 | +15% (−18 to +69) |
| RF C | 0.904 | 0.964 | 0.908 | 33 | −15% (−41 to +24) |
| L2-LR B | 0.905 | 0.968 | 0.911 | 30 | −23% (−49 to +14) |
| L2-LR C | 0.899 | 0.975 | 0.908 | 27 | −31% (−54 to +3) |

- Embeddings beat handcrafted features (RF B vs A: 36 fewer FP, CI −54 to −19). B to C helps the random forest (45 to 33; paired CI −20 to −4), not clearly L2-LR (−3; CI −9 to +3).
- **No model is distinguishable from MapBiomas** at the 95% level (all intervals above include 0 change), so treat reductions as promising, not confirmed.
- In RF C, permuting the embeddings costs 0.272 AUC and Oct-Nov SAG costs 0.012; radar, farm history, height and NDVI are about 0.
- VH is significant alone (OR 2.10 per SD for forest) and adjusted (1.99), but its importance is about 0 once embeddings are in (they reproduce 71% of VH). NDVI is not significant (p = 0.62).
- Sensitivity runs: swapping VV for VH (`run2d`) gives the same conclusions (L2-LR C with VV: 25 FP, CI of the change −58% to −6%, post hoc). Dropping height + farm history from C (`run2b`) changes FP by +2 (RF, CI −2 to +6): no detectable effect.

## Run
Input data is **not** committed (put it under `data/`, which is git-ignored): `cropin_model_features_2022_updated.csv`, `cropin_embeddings_2022.csv` (run 1: `cropin_model_features_2022.csv`).
```bash
pip install -r requirements.txt
cd run2_vh_embeddings
python scripts/poc2_vh_embeddings.py       # main run (~4 min on 1 CPU); writes results/ and figures
python scripts/run2b_drop_height_farm.py   # needs results/oof_predictions_run2.csv from the main run
python scripts/run2c_single_feature_checks.py
python scripts/run2d_vv_swap.py            # VV instead of VH; needs the main run's results
```
Each script's docstring explains its inputs; paths are relative to `run2_vh_embeddings/`.

## Caveats
Only 39 missed plantations (plantations MapBiomas calls forest), in 34 municipalities. One year (2022), one region, radar 2021-2022, canopy height 2020. Matching MapBiomas's forest recall accepts its ~10% forest loss, far from "technically zero". Bootstrap CIs do not include threshold-selection noise, and the VV swap was chosen after seeing the data. Not yet tested on Cropin's QC points. Features from MapBiomas Collection 9 (`ever_farm`) are not independent of the baseline. Saved run 1 models are fit on all rows: judge them by the out-of-fold results, not by in-sample scores.
