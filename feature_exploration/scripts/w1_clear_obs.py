"""Rule 13 – AUC of each passing feature by tercile of clear Sentinel-2 images (summer composite)."""
import pandas as pd
from sklearn.metrics import roc_auc_score
import w1_common as c
S = pd.read_csv('w1_screen.csv').set_index('feature'); mb, qc = c.load()
tr = mb[mb.agree >= 2].copy(); tr['ter'] = pd.qcut(tr.su_n_clear, 3, labels=['low (1-9)', 'mid (10-16)', 'high (17+)'])
rows = []
for f in S[S.PASS].index:
    sg = 1 if S.direction[f].startswith('higher') else -1; r = {'feature': f}
    for t, g in tr.groupby('ter', observed=True):
        g = g[g[f].notna()]; r[str(t)] = roc_auc_score(g.label, sg * g[f])
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv('w1_clear_obs_terciles.csv', index=False); print(T.round(2).to_string())
