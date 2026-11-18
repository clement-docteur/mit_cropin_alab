"""Master feature table v3 (13 Nov 2026): exactly the data the combined-model search used.
v2 + mm_B6_win filled (batch M1) + rebuilt harvest-cycle features (rh_, our Landsat) + land history (lh_, batch M2)
+ our Hansen tree cover 2000. Teammates' data columns (hc_, r2_) removed: their ideas are now rebuilt on our data."""
import numpy as np, pandas as pd
exec(open('/home/claude/combined/search_v2.py').read().split("def lr(C")[0])   # builds M (with AlphaEarth) and the pool P
META = ['uid', 'id', 'set', 'split', 'label', 'train_eligible', 'block', 'asof_year', 'agree', 'border', 'state', 'lon', 'lat',
        'area_ha', 'clear_month', 'tree_ndvi_summer', 'tree_filter_pass', 'team_pid', 'nc_wc_dist_nontree_m']
BENCH = ['nc_A_SAG', 'nc_chm_mean_r15', 'nc_chm_stdDev_r15']
V3 = M[META + P + BENCH + ['rh_forest_since_1985']]
OUT = '/home/claude/combined/v3/'; import os; os.makedirs(OUT, exist_ok=True)
V3.to_csv(OUT + 'master_features_v3.csv', index=False)
M[['uid', 'id', 'set', 'split'] + AE].to_csv(OUT + 'master_alphaearth_v3.csv', index=False)
from sklearn.metrics import roc_auc_score
def auc(c):
    x = T[[c, 'label']].dropna(); a = roc_auc_score(x.label, x[c]); return round(max(a, 1 - a), 2)
roles = [(c, 'feature') for c in P] + [(c, 'benchmark') for c in BENCH] + [('rh_forest_since_1985', 'veto (tested, did not help)')]
D = pd.DataFrame(roles, columns=['column', 'role']); D['auc_mb_train'] = D.column.map(auc)
D['missing_train'] = D.column.map(lambda c: int(T[c].isna().sum())); D['missing_qc'] = D.column.map(lambda c: int(M[M.set == 'qc'][c].isna().sum()))
D.to_csv(OUT + 'v3_columns.csv', index=False)
print(V3.shape, len(P)); print(D.to_string(index=False))
