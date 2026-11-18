"""Extras pre-declared in the protocol: finalists without MapBiomas-derived features; veto rule; plus a descriptive
comparison at equal QC forest errors (not used for selection)."""
import numpy as np, pandas as pd, sys
sys.argv = ['x']; exec(open('search_v1.py').read().split("edge = (T.nc_wc_dist")[0])
from sklearn.model_selection import GroupKFold
out = []
def evaluate(name, mid, pool, veto=False):
    X = T[pool]; s = np.full(len(T), np.nan)
    MODELS[mid] = (pool,) + MODELS[mid][1:]
    for tr, te in GroupKFold(5).split(X, y, g):
        m, _ = fit_model(mid, X.iloc[tr], y[tr], g[tr]); s[te] = m.predict_proba(X.iloc[te])[:, 1]
    m, _ = fit_model(mid, X, y, g); sq = m.predict_proba(Q[pool])[:, 1]
    if veto:
        s = np.where(T.rh_forest_since_1985.values == 1, 0, s); sq = np.where(Q.rh_forest_since_1985.values == 1, 0, sq)
    t = thr(y, s); fq = int((sq[yq == 0] > t).sum())
    out.append({'variant': name, 'MB AUC': round(roc_auc_score(y, s), 3), 'MB recall@5%': round(float((s[y == 1] > t).mean()), 3),
                'QC AUC': round(roc_auc_score(yq, sq), 3), 'QC FE': f'{fq}/58', 'QC recall': round(float((sq[yq == 1] > t).mean()), 3)})
    print(out[-1], flush=True); return sq
noMB = [c for c in P if c != 'nc_H_mapbiomas10m_plantation']
for mid, pool in (('I1', P), ('B5', PBB)):
    evaluate(f'{mid} without MapBiomas-derived', mid, [c for c in pool if c != 'nc_H_mapbiomas10m_plantation'])
    evaluate(f'{mid} + veto (forest since 1985)', mid, pool, veto=True)
# descriptive: recall at the QC threshold giving the same number of forest errors as REF (10/58)
qc = pd.read_csv('search_v1_qc.csv')
for mid in ['REF', 'I1', 'I2', 'I3', 'B1', 'B5', 'B6']:
    s = qc[mid].values; f = np.sort(s[yq == 0])[::-1]
    for k in (3, 6, 10):
        t = f[k]; out.append({'variant': f'{mid}: QC recall when QC forest errors fixed at {k}/58 (descriptive)', 'QC recall': round(float((s[yq == 1] > t).mean()), 3)})
pd.DataFrame(out).to_csv('search_v1_extra.csv', index=False); print(pd.DataFrame(out).to_string(index=False))
