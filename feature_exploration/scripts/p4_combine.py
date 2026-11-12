"""Phase 4 – combine families (decision 6 Nov: option 1 = Spectral A + Seasonal B, Structure E as add-on).
Pool = baseline features + A and B features meeting decision-rule criteria 1–3 (not context-dependent).
Stability selection (L1 logistic, 200 block-half subsamples), cap 10 features. Interpretable models only:
logistic regression and a depth-3 decision tree. Safety check on disturbed natural forest."""
import numpy as np, pandas as pd, json, sys
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import p4_common as c
mb, qc = c.load(); mb, info = c.alert_weights(mb, qc); trm = (mb.agree >= 2).values; y = mb.label.values
S = pd.concat([pd.read_csv('/home/claude/w1/w1_screen.csv'), pd.read_csv('/home/claude/w2/w2_screen.csv')])
ok = S.c1_q & S.c2_auc & S.c3_direction & ~S.context_dependent
pool = list(dict.fromkeys(c.BASE + S[ok & S.family.isin(['A', 'B'])].feature.tolist()))
tr = mb[trm]
# decorrelate (|Spearman| > 0.9 → keep the higher single-feature AUC)
auc1 = {f: max(roc_auc_score(tr.label, tr[f].fillna(tr[f].median())), 1 - roc_auc_score(tr.label, tr[f].fillna(tr[f].median()))) for f in pool}
C = tr[pool].corr(method='spearman').abs(); keep = []
for f in sorted(pool, key=lambda f: -auc1[f]):
    if all(C.loc[f, k] <= 0.9 for k in keep): keep.append(f)
rng = np.random.default_rng(2026); bl = tr.block.unique(); cnt = pd.Series(0.0, index=keep)
for _ in range(200):
    sub = tr[tr.block.isin(rng.choice(bl, len(bl) // 2, replace=False))]
    m = c.lr(C=0.05, penalty='l1').fit(sub[keep], sub.label); cnt += (np.abs(m[-1].coef_[0]) > 1e-6)
freq = (cnt / 200).sort_values(ascending=False)
sel = list(freq[freq >= 0.6].index[:10])
print('pool', len(pool), 'after decorrelation', len(keep)); print('selection frequency:', freq[freq > 0].round(2).to_dict()); print('selected:', sel)
dist = ((mb.label == 0) & (mb.D_n_events > 0)).values
out, scores = [], {}
def run(name, cols):
    r, s = c.evaluate(mb, qc, cols, name); t = c.thr(y, s, 0.05)
    r['FE_disturbed_forest'] = (s[dist] > t).mean(); r['n_disturbed_forest'] = int(dist.sum())
    out.append(r); scores[name] = s; print('done', name, file=sys.stderr)
run('Baseline (decision rule)', c.BASE)
run('Comparison: baseline + NFW', c.BASE_NFW)
run('P4 main: stability-selected A + B (logistic)', sel)
run('P4 + E add-on (ETH height, ETH model uncertainty)', sel + ['E_eth_h', 'E_eth_model_sd'])
run('P4 + D history (for reference only; not in pool)', sel + ['D_max_nbr_drop', 'D_cc_n_breaks', 'D_years_continuous_forest', 'D_h_total_loss_px', 'D_ndvi_1985_89', 'D_lt_n_vertices'])
# depth-3 tree on the same selected features (interpretable alternative)
def tree_oof(cols):
    s = np.full(len(mb), np.nan)
    for a, b in GroupKFold(5).split(mb, y, mb.block):
        a = a[trm[a]]; t = make_pipeline(SimpleImputer(strategy='median'), DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0))
        s[b] = t.fit(mb.iloc[a][cols], y[a]).predict_proba(mb.iloc[b][cols])[:, 1]
    return s
s = tree_oof(sel); t = c.thr(y, s, 0.05)
full = make_pipeline(SimpleImputer(strategy='median'), DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0)).fit(mb[trm][sel], y[trm])
q = full.predict_proba(qc[sel])[:, 1]
out.append(dict(model='P4 depth-3 tree (same features)', MB_AUC=roc_auc_score(y, s), **{f'MB_rec@{int(fe*100)}%': c.rec_at(y, s, fe) for fe in c.FES},
                QC_AUC=roc_auc_score(qc.label, q), QC_FE_at_MB_t=(q[qc.label == 0] > t).mean(), QC_rec_at_MB_t=(q[qc.label == 1] > t).mean(),
                FE_disturbed_forest=(s[dist] > t).mean()))
tree_txt = export_text(full[-1], feature_names=sel)
# final logistic coefficients (standardized) for interpretation
fm = c.lr().fit(mb[trm][sel], y[trm]); coefs = dict(zip(sel, fm[-1].coef_[0].round(3)))
# nested check of the whole Phase 4 selection (selection redone inside each outer fold)
oofn = np.full(len(mb), np.nan)
for a, b in GroupKFold(5).split(mb, y, mb.block):
    a2 = a[trm[a]]; t2 = mb.iloc[a2]; bl2 = t2.block.unique(); cn = pd.Series(0.0, index=keep)
    for _ in range(100):
        sub = t2[t2.block.isin(rng.choice(bl2, len(bl2) // 2, replace=False))]
        m = c.lr(C=0.05, penalty='l1').fit(sub[keep], sub.label); cn += (np.abs(m[-1].coef_[0]) > 1e-6)
    fq = (cn / 100).sort_values(ascending=False); sl = list(fq[fq >= 0.6].index[:10])
    oofn[b] = c.lr().fit(t2[sl], t2.label).predict_proba(mb.iloc[b][sl])[:, 1]
nested = dict(MB_AUC=roc_auc_score(y, oofn), **{f'MB_rec@{int(fe*100)}%': c.rec_at(y, oofn, fe) for fe in c.FES})
R = pd.DataFrame(out); R.to_csv('p4_models.csv', index=False)
json.dump(dict(pool=pool, decorrelated=keep, freq=freq.round(3).to_dict(), selected=sel, coefs=coefs, tree=tree_txt, nested=nested,
               alert_weights={str(k): v for k, v in info.items()}), open('p4_selection.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 320)
print(R[['model', 'MB_AUC', 'MB_rec@1%', 'MB_rec@2%', 'MB_rec@5%', 'QC_AUC', 'QC_AUC_lo', 'QC_AUC_hi', 'QC_FE_at_MB_t', 'QC_rec_at_MB_t', 'QC_FE_at_weighted_t', 'QC_rec_at_weighted_t', 'FE_disturbed_forest', 'grp_border_FE', 'grp_agree2_FE']].round(3).to_string())
print('coefs', coefs); print(tree_txt); print('nested', {k: round(v, 3) for k, v in nested.items()})
