"""Single scoring of the finalists on the QC held-back half (protocol + addendum v2 + finalists record)."""
import numpy as np, pandas as pd, json
exec(open('search_v2.py').read().split("rows, oofs, qcs = [], {}, {}")[0])
H = M[(M.set == 'qc') & (M.split == 'heldback')].reset_index(drop=True); yh = H.label.values; gh = H.block.values
MBD = ['nc_H_mapbiomas10m_plantation', 'lh_natural_run_years', 'lh_n_changes', 'lh_ever_open']
def run(name, mid, pool):
    if mid == 'REF':
        so = T.nc_A_SAG.values.astype(float); sq = Q.nc_A_SAG.values.astype(float); sh = H.nc_A_SAG.values.astype(float)
    else:
        MODELS[mid] = (pool,) + MODELS[mid][1:]; X = T[pool]; so = np.full(len(T), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            m, _ = fit_model(mid, X.iloc[tr], y[tr], g[tr]); so[te] = m.predict_proba(X.iloc[te])[:, 1]
        m, _ = fit_model(mid, X, y, g); sq = m.predict_proba(Q[pool])[:, 1]; sh = m.predict_proba(H[pool])[:, 1]
    t, tw = thr(y, so), wthr(y, so, w)
    out = {'model': name, 'MB OOF AUC': round(roc_auc_score(y, so), 3), 'QC dev AUC': round(roc_auc_score(yq, sq), 3),
           'QC dev FE': f'{int((sq[yq == 0] > t).sum())}/{int((yq == 0).sum())}', 'QC dev recall': round(float((sq[yq == 1] > t).mean()), 3)}
    lo, hi = boot_auc(yh, sh, gh); nf = int((yh == 0).sum()); fe = int((sh[yh == 0] > t).sum()); few = int((sh[yh == 0] > tw).sum())
    edge_h = (H.nc_wc_dist_nontree_m <= 10).values
    out.update({'HELD-BACK AUC': f'{roc_auc_score(yh, sh):.3f} [{lo:.2f}–{hi:.2f}]', 'HELD-BACK forest errors': f'{fe}/{nf} ({wilson(fe, nf)})',
                'HELD-BACK recall': round(float((sh[yh == 1] > t).mean()), 3),
                'HELD-BACK FE (alert-weighted t)': f'{few}/{nf}', 'HELD-BACK recall (alert-weighted t)': round(float((sh[yh == 1] > tw).mean()), 3),
                'HELD-BACK edge FE (≤10 m)': f'{int((sh[(yh == 0) & edge_h] > t).sum())}/{int(((yh == 0) & edge_h).sum())}'})
    return out, sh, t
res, S, TH = [], {}, {}
for name, mid, pool in (('REF: SAG', 'REF', None), ('I2: sparse logistic regression', 'I2', P), ('B5: random forest + AlphaEarth', 'B5', PBB),
                        ('I2 without MapBiomas-derived (secondary)', 'I2', [c for c in P if c not in MBD]),
                        ('B5 without MapBiomas-derived (secondary)', 'B5', [c for c in PBB if c not in MBD])):
    r, sh, t = run(name, mid, pool); res.append(r); S[name], TH[name] = sh, t; print(r, flush=True)
# paired comparisons vs SAG on held-back
bl = np.unique(gh); idx = {b: np.where(gh == b)[0] for b in bl}; rng = np.random.default_rng(2026); comp = []
ref = S['REF: SAG']; f0 = yh == 0; eR = ref[f0] > TH['REF: SAG']
for name in list(S)[1:]:
    d = []
    for _ in range(2000):
        ii = np.concatenate([idx[b] for b in rng.choice(bl, len(bl))])
        if len(set(yh[ii])) == 2: d.append(roc_auc_score(yh[ii], S[name][ii]) - roc_auc_score(yh[ii], ref[ii]))
    d = np.array(d); eM = S[name][f0] > TH[name]; fixed, new = int((eR & ~eM).sum()), int((eM & ~eR).sum())
    comp.append({'model vs SAG': name, 'AUC diff': round(roc_auc_score(yh, S[name]) - roc_auc_score(yh, ref), 3),
                 '95% CI': f'{np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f}', 'one-sided p': round(float((d <= 0).mean()), 4),
                 'forest errors fixed / new': f'{fixed} / {new}', 'McNemar p': round(binomtest(min(fixed, new), fixed + new, 0.5).pvalue, 3) if fixed + new else 1.0})
pd.DataFrame(res).to_csv('heldback_v2_results.csv', index=False); pd.DataFrame(comp).to_csv('heldback_v2_comparisons.csv', index=False)
pd.set_option('display.width', 250); print(pd.DataFrame(res).to_string(index=False)); print(pd.DataFrame(comp).to_string(index=False))
