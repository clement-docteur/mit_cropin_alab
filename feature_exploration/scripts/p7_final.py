"""Phase 7 – final test on the sealed sets, exactly as locked in 'Phase 7 Protocol' (11 Nov 2026).
Models are fitted on MapBiomas dev only (same specification as when the thresholds were locked) and the
coefficients are checked against p7_locked_thresholds.json before anything sealed is scored."""
import numpy as np, pandas as pd, json
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import p7_common as c
LOCK = json.load(open('/home/claude/p6/p7_locked_thresholds.json'))
dev, qcdev = c.load(); ydev = dev.label.values; trm = (dev.agree >= 2).values
lb = pd.read_csv('sealed/w3_mb_dev_features.csv'); hq = pd.read_csv('sealed/w3_qc_dev_features.csv')
n_lb_raw = len(lb); lb = lb[lb.A_su_NDVI > 0.6].reset_index(drop=True)
fh = pd.read_csv('/mnt/user-data/uploads/Cropin - A Lab/Feature Exploration/results/phase7/lockbox_forest_history.csv')[['id', 'forest_all_1985_2012']]
lb = lb.merge(fh, on='id', how='left')
CAND = {'Lean 4 (main)': ('Lean 4', ['B_SAG_phase_sin', 'A_su_NDMI', 'B_SAG_SON', 'A_sg_NDVI']),
        'P4 (challenger)': ('P4', ['B_SAG_phase_sin', 'A_su_NDMI', 'B_NDRE_phase_sin', 'B_SAG_SON', 'A_sg_NDVI', 'B_SWIR_phase_sin', 'B_SAG_mean', 'B_SWIR_phase_cos', 'B_SAG_DJ', 'A_SAG']),
        'Baseline': ('Baseline', c.BASE), 'Baseline + NFW': ('Baseline + NFW', c.BASE_NFW)}
def wil(k, n):
    if n == 0: return (np.nan, np.nan)
    p = k / n; z = 1.96; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, m - h), min(1, m + h))
models, S = {}, {}
for name, (key, cols) in CAND.items():
    m = c.lr().fit(dev[trm][cols], ydev[trm]); coefs = dict(zip(cols, m[-1].coef_[0]))
    assert all(abs(coefs[k] - v) < 1e-3 for k, v in LOCK[key]['coefs_std'].items()), f'coefficients differ from locked for {name}'
    models[name] = (m, cols, LOCK[key])
# AlphaEarth benchmark (H28): logistic on 64 embedding bands, MapBiomas dev only
AE = pd.read_csv('/mnt/user-data/uploads/cropin_texture/p7_alphaearth.csv'); ae_cols = [f'A{i:02d}' for i in range(64)]
AE = AE[['id', 'set'] + ae_cols].rename(columns={k: 'AE_' + k for k in ae_cols}); ae_cols = ['AE_' + k for k in ae_cols]
dev = dev.merge(AE, on=['id', 'set'], how='left'); lb = lb.merge(AE, on=['id', 'set'], how='left'); hq = hq.merge(AE, on=['id', 'set'], how='left')
s_ae = c.oof(dev, ae_cols, trm)
ae_lock = {f'FE{int(fe*100)}': c.thr(ydev, s_ae, fe) for fe in (0.01, 0.02, 0.05)}
models['AlphaEarth + logistic (benchmark)'] = (c.lr().fit(dev[trm][ae_cols], ydev[trm]), ae_cols, ae_lock)
def evaluate(d, setname):
    y = d.label.values; g = d.block.values; rows = []
    for name, (m, cols, L) in models.items():
        s = m.predict_proba(d[cols])[:, 1]; S[(setname, name)] = s
        auc = roc_auc_score(y, s); lo, hi = c.bci(y, s, g, roc_auc_score, 1000)
        r = dict(set=setname, model=name, n_forest=int((y == 0).sum()), n_plant=int((y == 1).sum()), AUC=auc, AUC_CI=f'{lo:.3f}–{hi:.3f}')
        for t in ('FE5', 'FE2', 'FE1', 'FE5_alert_weighted'):
            if t not in L: continue
            k = int((s[y == 0] > L[t]).sum()); n = int((y == 0).sum()); a, b = wil(k, n)
            r[f'{t} forest error'] = f'{k}/{n} = {k/n:.3f} ({a:.2f}–{b:.2f})'; r[f'{t} recall'] = round((s[y == 1] > L[t]).mean(), 3)
        rows.append(r)
    # NFW thresholds, MapBiomas map, JRC GFT (binary benchmarks)
    for t in (0.30, 0.50, 0.52, 0.70):
        call = d.nfw.values < t * 250; k = int(call[y == 0].sum()); n = int((y == 0).sum()); a, b = wil(k, n)
        rows.append(dict(set=setname, model=f'NFW alone, threshold {t} (B0 < {t*250:.0f} = plantation)', n_forest=n, n_plant=int((y == 1).sum()),
                         AUC=roc_auc_score(y, -d.nfw.values), **{'FE5 forest error': f'{k}/{n} = {k/n:.3f} ({a:.2f}–{b:.2f})', 'FE5 recall': round(call[y == 1].mean(), 3)}))
    for col, nm in (('H_mapbiomas10m_plantation', 'MapBiomas 10 m map (plantation class)'), ('H_gft_planted', 'JRC forest types 2020 ("planted")')):
        call = d[col].values == 1; k = int(call[y == 0].sum()); n = int((y == 0).sum()); a, b = wil(k, n)
        rows.append(dict(set=setname, model=nm, n_forest=n, n_plant=int((y == 1).sum()), AUC=roc_auc_score(y, d[col].values),
                         **{'FE5 forest error': f'{k}/{n} = {k/n:.3f} ({a:.2f}–{b:.2f})', 'FE5 recall': round(call[y == 1].mean(), 3)}))
    return rows
R = pd.DataFrame(evaluate(hq, 'QC held-back') + evaluate(lb, 'MapBiomas lockbox'))
# pre-declared comparisons
C = []
for setname, d in (('QC held-back', hq), ('MapBiomas lockbox', lb)):
    y = d.label.values; g = d.block.values; bl = np.unique(g); idx = {b: np.where(g == b)[0] for b in bl}; rng = np.random.default_rng(2026)
    sb = S[(setname, 'Baseline')]; tb = LOCK['Baseline']['FE5']
    for name in ('Lean 4 (main)', 'P4 (challenger)'):
        s = S[(setname, name)]; t = models[name][2]['FE5']; diffs = []
        for _ in range(2000):
            ii = np.concatenate([idx[b] for b in rng.choice(bl, len(bl))])
            if len(set(y[ii])) == 2: diffs.append(roc_auc_score(y[ii], s[ii]) - roc_auc_score(y[ii], sb[ii]))
        diffs = np.array(diffs); d_auc = roc_auc_score(y, s) - roc_auc_score(y, sb)
        f = y == 0; e_m = s[f] > t; e_b = sb[f] > tb; b_ = int((e_b & ~e_m).sum()); c_ = int((e_m & ~e_b).sum())
        p_mc = binomtest(c_, b_ + c_, 0.5, alternative='less').pvalue if b_ + c_ > 0 else 1.0   # H1: model makes fewer forest errors
        C.append(dict(set=setname, comparison=f'{name} vs Baseline', AUC_difference=round(d_auc, 3), AUC_diff_CI=f'{np.percentile(diffs, 2.5):.3f}–{np.percentile(diffs, 97.5):.3f}',
                      p_one_sided_AUC=round((diffs <= 0).mean(), 4), forest_errors_model=int(e_m.sum()), forest_errors_baseline=int(e_b.sum()),
                      fixed_by_model=b_, new_errors_by_model=c_, McNemar_exact_p_one_sided=round(p_mc, 4), n_forest=int(f.sum())))
CP = pd.DataFrame(C)
# groups on the lockbox (Phase 6 groups) at the primary threshold
y = lb.label.values
G = {'All': np.ones(len(lb), bool), 'Within 10 m of non-tree (edge)': lb.wc_dist_nontree_m <= 10, '10–30 m': lb.wc_dist_nontree_m.between(10.01, 30),
     '> 30 m': lb.wc_dist_nontree_m > 30, 'Border flag = 1': lb.border == 1, 'Agreement 3/3': lb.agree == 3, 'Agreement 2/3': lb.agree == 2,
     'Agreement 1/3': lb.agree == 1, 'PR': lb.state == 'PR', 'SC': lb.state == 'SC', 'RS': lb.state == 'RS',
     'Younger forest (HC2 proxy)': (lb.forest_all_1985_2012 == 0) | (lb.label == 1), 'Disturbed natural forest': (lb.D_n_events > 0) | (lb.label == 1)}
GR = []
for gname, m in G.items():
    m = np.asarray(m); r = dict(group=gname, n_forest=int(((y == 0) & m).sum()), n_plant=int(((y == 1) & m).sum()))
    for name in ('Lean 4 (main)', 'P4 (challenger)', 'Baseline'):
        s = S[('MapBiomas lockbox', name)]; t = models[name][2]['FE5']; k = int((s[m & (y == 0)] > t).sum()); n = r['n_forest']
        r[f'{name} FE'] = f'{k}/{n}' if n else ''; r[f'{name} recall'] = round((s[m & (y == 1)] > t).mean(), 2) if r['n_plant'] else np.nan
    GR.append(r)
GR = pd.DataFrame(GR)
for nm, df in (('p7_results', R), ('p7_comparisons', CP), ('p7_lockbox_groups', GR)): df.to_csv(f'{nm}.csv', index=False)
pd.set_option('display.width', 330); pd.set_option('display.max_colwidth', 60); pd.set_option('display.max_rows', 60)
print('lockbox', n_lb_raw, '->', len(lb), 'after tree filter; QC held-back', len(hq))
print(R.round(3).to_string()); print(CP.to_string()); print(GR.to_string())
