"""Wave 1 – single-feature screen with the plan's decision rule (§9), per family A, C, D."""
import numpy as np, pandas as pd, re, sys
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import w1_common as c
mb, qc = c.load(); mb, info = c.alert_weights(mb, qc)
tr = mb[mb.agree >= 2].reset_index(drop=True); trm = (mb.agree >= 2).values
def prior(f):   # +1 = hypothesis predicts higher in plantation, -1 = lower, 0 = no stated prior
    if f == 'A_SAG': return 1
    if re.match(r'A_(su|sg)_B11$', f): return -1
    if f.startswith('C_'):
        if re.search(r'_(contrast|ent|var|sd)_w', f) or f in ('C_dir_contrast_mean',) or f.startswith('C_seg_sd'): return -1
        if re.search(r'_(idm|asm)_w', f) or f == 'C_dir_anisotropy': return 1
        return 0
    P = {'D_n_events': 1, 'D_years_since_last_event': -1, 'D_mean_interval': -1, 'D_max_nbr_drop': 1, 'D_regrowth_3yr': 1,
         'D_years_continuous_forest': -1, 'D_ndvi_1985_89': -1, 'D_ndvi_sd_last10': 1, 'D_ndvi_absdiff_last10': 1,
         'D_lt_n_vertices': 1, 'D_lt_n_dist': 1, 'D_lt_greatest_mag': 1, 'D_lt_years_since_greatest': -1,
         'D_lt_greatest_duration': -1, 'D_cc_n_breaks': 1, 'D_cc_years_since_break': -1,
         'D_h_max_year_px': 1, 'D_h_n_loss_years': 1, 'D_h_total_loss_px': 1}
    return P.get(f, 0)
fams = {p: [x for x in mb.columns if x.startswith(p + '_') and x not in ('D_n_years_obs',)] for p in 'ACD'}
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
R.to_csv('w1_screen.csv', index=False)
pd.DataFrame([dict(base_rec1=base_rec[0.01], base_rec2=base_rec[0.02], base_rec5=base_rec[0.05])]).to_csv('w1_baseline_rec.csv', index=False)
print('baseline (SAG+height+height SD) OOF recall:', {k: round(v, 3) for k, v in base_rec.items()})
print(R.groupby('family')[['c1_q', 'c2_auc', 'c3_direction', 'c4_recall', 'PASS', 'context_dependent']].sum())
pd.set_option('display.width', 250)
print(R[R.PASS].sort_values(['family', 'd_rec5'], ascending=[True, False])[['family', 'feature', 'direction', 'AUC', 'AUC_lo', 'QC_AUC_raw', 'context_dependent', 'd_rec1', 'd_rec2', 'd_rec5']].round(3).to_string())
