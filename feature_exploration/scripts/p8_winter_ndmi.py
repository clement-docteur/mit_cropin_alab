"""Post-hoc test (locked 12 Nov 2026, Run Log): winter NDMI (Apr–Sep median of monthly NDMI, 3-year as-of window).
Trained on MapBiomas dev only; scored once on QC held-back and MapBiomas lockbox."""
import numpy as np, pandas as pd, json
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
import p8_common as c
S = pd.concat([pd.read_csv('/mnt/user-data/uploads/cropin_texture/w2_s2_monthly_mb.csv'), pd.read_csv('/mnt/user-data/uploads/cropin_texture/w2_s2_monthly_qc.csv')]).dropna(subset=['B8'])
S['NDMI'] = (S.B8 - S.B11) / (S.B8 + S.B11); S['m'] = pd.to_datetime(S.month).dt.month
W = S[S.m.between(4, 9)].groupby(['id', 'set']).NDMI.median().rename('W_NDMI').reset_index()
dev, _ = c.load(); dev = dev.merge(W, on=['id', 'set'], how='left')
lb = pd.read_csv('/home/claude/p7/sealed/w3_mb_dev_features.csv'); lb = lb[lb.A_su_NDVI > 0.6].reset_index(drop=True).merge(W, on=['id', 'set'], how='left')
hq = pd.read_csv('/home/claude/p7/sealed/w3_qc_dev_features.csv').merge(W, on=['id', 'set'], how='left')
y = dev.label.values; trm = (dev.agree >= 2).values
L4 = ['B_SAG_phase_sin', 'A_su_NDMI', 'B_SAG_SON', 'A_sg_NDVI']; L4W = L4 + ['W_NDMI']
LOCK = json.load(open('/home/claude/p6/p7_locked_thresholds.json'))
t_L4 = LOCK['Lean 4']['FE5']; s_dev = c.oof(dev, L4W, trm); t_L4W = c.thr(y, s_dev, 0.05)
mL4 = c.lr().fit(dev[trm][L4], y[trm]); mL4W = c.lr().fit(dev[trm][L4W], y[trm])
def wil(k, n):
    p = k / n; z = 1.96; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f'{max(0, m - h):.2f}–{min(1, m + h):.2f}'
out = {'missing W_NDMI': {'dev': int(dev.W_NDMI.isna().sum()), 'QC held-back': int(hq.W_NDMI.isna().sum()), 'lockbox': int(lb.W_NDMI.isna().sum())},
       'threshold_L4W_FE5_dev': float(t_L4W), 'dev_OOF_AUC_L4W': float(roc_auc_score(y, s_dev))}
rng = np.random.default_rng(2026)
for name, d in (('QC held-back', hq), ('MapBiomas lockbox', lb)):
    yy = d.label.values; g = d.block.values; bl = np.unique(g); idx = {b: np.where(g == b)[0] for b in bl}
    r = {}
    for f in ('W_NDMI', 'A_su_NDMI'):
        r[f'AUC alone {f}'] = round(roc_auc_score(yy, d[f].fillna(d[f].median())), 3)
    a = mL4.predict_proba(d[L4])[:, 1]; b = mL4W.predict_proba(d[L4W])[:, 1]
    diffs = []
    for _ in range(2000):
        ii = np.concatenate([idx[k] for k in rng.choice(bl, len(bl))])
        if len(set(yy[ii])) == 2: diffs.append(roc_auc_score(yy[ii], b[ii]) - roc_auc_score(yy[ii], a[ii]))
    diffs = np.array(diffs); f0 = yy == 0
    eA = a[f0] > t_L4; eB = b[f0] > t_L4W; fixed = int((eA & ~eB).sum()); new = int((eB & ~eA).sum())
    r.update({'AUC Lean 4': round(roc_auc_score(yy, a), 3), 'AUC Lean 4 + winter NDMI': round(roc_auc_score(yy, b), 3),
              'AUC difference': round(roc_auc_score(yy, b) - roc_auc_score(yy, a), 3), 'diff 95% CI': f'{np.percentile(diffs, 2.5):.3f}–{np.percentile(diffs, 97.5):.3f}',
              'one-sided p': round(float((diffs <= 0).mean()), 4),
              'forest errors Lean 4': f'{int(eA.sum())}/{int(f0.sum())} ({wil(int(eA.sum()), int(f0.sum()))})',
              'forest errors Lean 4 + W': f'{int(eB.sum())}/{int(f0.sum())} ({wil(int(eB.sum()), int(f0.sum()))})',
              'fixed / new forest errors': f'{fixed} / {new}', 'McNemar p (fewer errors)': round(binomtest(new, fixed + new, 0.5, alternative="less").pvalue, 3) if fixed + new else 1.0,
              'recall Lean 4': round(float((a[~f0] > t_L4).mean()), 3), 'recall Lean 4 + W': round(float((b[~f0] > t_L4W).mean()), 3)})
    out[name] = r
json.dump(out, open('p8_winter_ndmi_results.json', 'w'), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
