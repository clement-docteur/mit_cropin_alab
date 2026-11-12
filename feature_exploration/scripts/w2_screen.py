"""Wave 2 – single-feature screen with the plan's decision rule (§9), per family B, F."""
import numpy as np, pandas as pd, re, sys
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import w2_common as c
mb, qc = c.load(); mb, info = c.alert_weights(mb, qc)
tr = mb[mb.agree >= 2].reset_index(drop=True); trm = (mb.agree >= 2).values
def prior(f):   # pre-registered in Run Log (2026-11-06) before any Wave 2 data were seen
    return {'B_NDVI_amp': -1, 'B_NDVI_drought_anom': 1, 'B_NDMI_drought_anom': 1}.get(f, 0)
fams = {p: [x for x in mb.columns if x.startswith(p + '_') and x not in ('B_n_months', 'B_n_drought_months')] for p in 'BF'}
base_s = c.oof(mb, c.BASE, trm); y = mb.label.values
base_rec = {fe: c.rec_at(y, base_s, fe) for fe in c.FES}
rows = []
for fam, feats in fams.items():
    for f in feats:
        d = tr[tr[f].notna()]; x = d[f].values; yy = d.label.values
        a = roc_auc_score(yy, x); sg = 1 if a >= 0.5 else -1
        lo, hi = c.bci(yy, sg * x, d.block.values, roc_auc_score, 300)
        p = mannwhitneyu(x[yy == 1], x[yy == 0]).pvalue
        st = {s: roc_auc_score(g.label, g[f]) for s, g in d.groupby('state') if g.label.nunique() == 2}
        state_ok = all((v - 0.5) * sg > 0 for v in st.values())
        q = qc[qc[f].notna()]; qa = roc_auc_score(q.label, q[f]) if len(q) > 20 else np.nan
        ctxdep = (not np.isnan(qa)) and ((qa - 0.5) * sg < -0.05)
        s = c.oof(mb, list(dict.fromkeys(c.BASE + [f])), trm)
        dr = {fe: c.rec_at(y, s, fe) - base_rec[fe] for fe in c.FES}
        pr = prior(f)
        rows.append(dict(family=fam, feature=f, n=len(d), prior=pr, direction='higher = plantation' if sg == 1 else 'lower = plantation',
                         AUC=max(a, 1 - a), AUC_lo=lo, AUC_hi=hi, MW_p=p, state_AUCs='; '.join(f'{k} {v:.2f}' for k, v in st.items()),
                         state_consistent=state_ok, prior_match=(pr == 0) or (pr == sg), QC_AUC_raw=qa, context_dependent=ctxdep,
                         d_rec1=dr[0.01], d_rec2=dr[0.02], d_rec5=dr[0.05]))
    print(fam, 'done', file=sys.stderr)
R = pd.DataFrame(rows)
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out
R['q_BH'] = R.groupby('family').MW_p.transform(bh)
R['c1_q'] = R.q_BH < 0.05; R['c2_auc'] = R.AUC_lo > 0.60; R['c3_direction'] = R.state_consistent & R.prior_match
R['c4_recall'] = (R.d_rec5 >= 0.05) & (R.d_rec1 >= 0) & (R.d_rec2 >= 0)
R['PASS'] = R.c1_q & R.c2_auc & R.c3_direction & R.c4_recall
R.to_csv('w2_screen.csv', index=False)
pd.DataFrame([dict(base_rec1=base_rec[0.01], base_rec2=base_rec[0.02], base_rec5=base_rec[0.05])]).to_csv('w2_baseline_rec.csv', index=False)
print('baseline (SAG+height+height SD) OOF recall:', {k: round(v, 3) for k, v in base_rec.items()})
print(R.groupby('family')[['c1_q', 'c2_auc', 'c3_direction', 'c4_recall', 'PASS', 'context_dependent']].sum())
pd.set_option('display.width', 250)
print(R[R.PASS].sort_values(['family', 'd_rec5'], ascending=[True, False])[['family', 'feature', 'direction', 'AUC', 'AUC_lo', 'QC_AUC_raw', 'context_dependent', 'd_rec1', 'd_rec2', 'd_rec5']].round(3).to_string())
