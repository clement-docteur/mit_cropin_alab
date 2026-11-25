"""Master v34, step 8: master_features_v34_slim.csv = master_features_v3_slim.csv (unchanged, all 2,667 rows)
+ the November QC polygons that are NOT repeats of v3 polygons (split = 'nov', sealed blind test: never tune on it).
Also writes master_alphaearth_v34.csv (v3 AlphaEarth rows + the same new rows).
Repeats of v3 polygons (split 'nov_repeat') are left out because their v3 rows are already in the table;
their features are kept in 'Master Data/v34_work/qcnov_features.csv'.
Run after 05/06 (and again after round 3 to fill nc_D_h_total_loss_px and mm_B6_win)."""
import json, os, subprocess, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = f'{HERE}/..'; W = f'{ROOT}/Master Data/v34_work'
L = json.load(open(f'{W}/rh_levels.json'))
subprocess.run([sys.executable, f'{HERE}/05_build_features.py', str(L['closed_canopy_nbr']), str(L['open_land_nbr'])], check=True)

v3 = pd.read_csv(f'{ROOT}/Master Data/master_features_v3_slim.csv')
F = pd.read_csv(f'{W}/qcnov_features.csv')
F = F[F.split == 'nov'].copy()
F['label_forest'] = 1 - F.label
F['train_eligible'] = False
F['border'] = np.nan
new = F.reindex(columns=v3.columns)
missing = [c for c in v3.columns if c not in F.columns]
assert not missing, f'columns not built: {missing}'
assert not set(new.uid) & set(v3.uid)
out = pd.concat([v3, new], ignore_index=True)
for c in v3.columns:   # keep v3 dtypes where the new rows allow it
    if v3[c].dtype.kind in 'iub' and out[c].notna().all(): out[c] = out[c].astype(v3[c].dtype)
out.to_csv(f'{ROOT}/Master Data/master_features_v34_slim.csv', index=False)

A3 = pd.read_csv(f'{ROOT}/Master Data/master_alphaearth_v3.csv')
AE = pd.read_csv(f'{W}/qcnov_alphaearth.csv').merge(F[['id', 'uid', 'set', 'split']], on='id')
pd.concat([A3, AE.reindex(columns=A3.columns)], ignore_index=True).to_csv(f'{ROOT}/Master Data/master_alphaearth_v34.csv', index=False)

print('v34 slim:', out.shape, '| v3 rows unchanged:', out.iloc[:len(v3)].equals(v3.astype(out.dtypes.to_dict())) or 'check dtypes')
print(out.groupby(['set', 'split']).size().to_string())
print('new rows by label:', new.label_forest.map({1: 'forest', 0: 'plantation'}).value_counts().to_dict())
na = new.isna().sum(); print('new rows, columns with missing values:', na[na > 0].to_dict())
