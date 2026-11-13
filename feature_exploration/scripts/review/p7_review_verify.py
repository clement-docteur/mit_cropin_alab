import numpy as np, pandas as pd, json, warnings
from scipy.stats import binomtest, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)

L = json.load(open('/home/claude/p6/p7_locked_thresholds.json'))
dev_raw = pd.read_csv('/home/claude/w3/w3_mb_dev_features.csv')
lb_raw = pd.read_csv('/home/claude/p7/sealed/w3_mb_dev_features.csv')
qc = pd.read_csv('/home/claude/p7/sealed/w3_qc_dev_features.csv')
AE = pd.read_csv('/mnt/user-data/uploads/cropin_texture/p7_alphaearth.csv')
dev = dev_raw[dev_raw.A_su_NDVI > 0.6].reset_index(drop=True)
lb = lb_raw[lb_raw.A_su_NDVI > 0.6].reset_index(drop=True)
print('dev raw', len(dev_raw), 'filtered', len(dev), 'train(agree>=2)', (dev.agree >= 2).sum())
print('lockbox raw', len(lb_raw), lb_raw.label.value_counts().to_dict(), '-> filtered', len(lb), lb.label.value_counts().to_dict())
print('lockbox dropped by filter:', lb_raw[lb_raw.A_su_NDVI <= 0.6][['id', 'label', 'A_su_NDVI', 'agree']].to_string())
print('lockbox NDVI NaN:', lb_raw.A_su_NDVI.isna().sum(), 'dev NDVI NaN', dev_raw.A_su_NDVI.isna().sum())
print('QC', len(qc), qc.label.value_counts().to_dict(), 'dup ids', qc.id.duplicated().sum(), 'label NaN', qc.label.isna().sum())

FEAT = {'Lean 4': ['B_SAG_phase_sin', 'A_su_NDMI', 'B_SAG_SON', 'A_sg_NDVI']}
FEAT['P4'] = FEAT['Lean 4'] + ['B_NDRE_phase_sin', 'B_SWIR_phase_sin', 'B_SAG_mean', 'B_SWIR_phase_cos', 'B_SAG_DJ', 'A_SAG']
FEAT['Baseline'] = ['A_SAG', 'chm_mean_r15', 'chm_stdDev_r15']
FEAT['Baseline + NFW'] = FEAT['Baseline'] + ['nfw']

def mk():
    return Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler()),
                     ('l', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=3000))])
tr = dev[dev.agree >= 2]
M, maxdiff = {}, {}
for k, cols in FEAT.items():
    m = mk().fit(tr[cols], tr.label); M[k] = m
    co = dict(zip(cols, m.named_steps['l'].coef_[0]))
    maxdiff[k] = max(max(abs(co[c] - v) for c, v in L[k]['coefs_std'].items()), abs(m.named_steps['l'].intercept_[0] - L[k]['intercept']))
    assert set(co) == set(L[k]['coefs_std'])
print('max |coef diff| vs lock (incl intercept):', maxdiff)

# Re-derive locked thresholds from dev OOF to confirm they are dev-based (method: 'higher' quantile of OOF forest scores)
def oof(d, cols):
    s = np.full(len(d), np.nan); tm = (d.agree >= 2).values
    for a, b in GroupKFold(5).split(d, d.label, d.block):
        a2 = a[tm[a]]; s[b] = mk().fit(d.iloc[a2][cols], d.iloc[a2].label).predict_proba(d.iloc[b][cols])[:, 1]
    return s
yd = dev.label.values
for k, cols in FEAT.items():
    s = oof(dev, cols); f = s[yd == 0]
    print('re-derived dev thresholds', k, {fe: round(np.quantile(f, 1 - fe, method='higher'), 4) for fe in (0.05, 0.02, 0.01)},
          'locked', {t: round(L[k][t], 4) for t in ('FE5', 'FE2', 'FE1')})

AEc = [f'A{i:02d}' for i in range(64)]
print('AE dup id+set', AE.duplicated(['id', 'set']).sum(), 'AE NaN rows', AE[AEc].isna().any(axis=1).sum())
ae = AE[['id', 'set', 'emb_year', 'cutoff'] + AEc].rename(columns={'cutoff': 'ae_cutoff'})
dev = dev.merge(ae, on=['id', 'set'], how='left'); lb = lb.merge(ae, on=['id', 'set'], how='left'); qc = qc.merge(ae, on=['id', 'set'], how='left')
for nm, d in (('dev', dev), ('lb', lb), ('qc', qc)):
    print(nm, 'AE missing', d[AEc[0]].isna().sum(), 'emb_year counts', d.emb_year.value_counts().to_dict(), 'cutoff match', (d.cutoff == d.ae_cutoff).mean())
s_ae = oof(dev, AEc); fa = s_ae[yd == 0]
aeL = {f'FE{int(fe*100)}': np.quantile(fa, 1 - fe, method='higher') for fe in (0.05, 0.02, 0.01)}
print('AE OOF dev AUC', round(roc_auc_score(yd, s_ae), 4), 'AE thresholds', aeL)
tr = dev[dev.agree >= 2]; M['AlphaEarth'] = mk().fit(tr[AEc], tr.label); FEAT['AlphaEarth'] = AEc

def wil(k, n):
    p = k / n; z = 1.96; d = 1 + z*z/n; m = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d; return max(0, m-h), min(1, m+h)
rows, S = [], {}
for setn, d in (('QC held-back', qc), ('MapBiomas lockbox', lb), ('Lockbox agree>=2', lb[lb.agree >= 2].reset_index(drop=True))):
    y = d.label.values
    for k in FEAT:
        s = M[k].predict_proba(d[FEAT[k]])[:, 1]; S[(setn, k)] = s
        r = dict(set=setn, model=k, nF=(y == 0).sum(), nP=(y == 1).sum(), AUC=round(roc_auc_score(y, s), 4))
        TH = L[k] if k in L else aeL
        for t in ('FE5', 'FE2', 'FE1', 'FE5_alert_weighted'):
            if t in TH:
                r[t+'_k'] = int((s[y == 0] > TH[t]).sum()); r[t+'_rec'] = round((s[y == 1] > TH[t]).mean(), 3)
        rows.append(r)
    for t in (0.30, 0.50, 0.52, 0.70):
        c = d.nfw.values < t * 250
        rows.append(dict(set=setn, model=f'NFW<{t}', nF=(y == 0).sum(), nP=(y == 1).sum(), AUC=round(roc_auc_score(y, -d.nfw.values), 4),
                         FE5_k=int(c[y == 0].sum()), FE5_rec=round(c[y == 1].mean(), 3)))
    for col in ('H_mapbiomas10m_plantation', 'H_gft_planted'):
        c = d[col].values == 1
        rows.append(dict(set=setn, model=col, nF=(y == 0).sum(), nP=(y == 1).sum(), AUC=round(roc_auc_score(y, d[col].values), 4),
                         FE5_k=int(c[y == 0].sum()), FE5_rec=round(c[y == 1].mean(), 3), nan=int(d[col].isna().sum())))
    print(setn, 'nfw NaN', d.nfw.isna().sum(), 'nfw range', d.nfw.min(), d.nfw.max(),
          'MB map vals', d.H_mapbiomas10m_plantation.value_counts(dropna=False).to_dict(), 'GFT vals', d.H_gft_planted.value_counts(dropna=False).to_dict())
R = pd.DataFrame(rows); print(R.to_string())
R.to_csv('/tmp/claude-0/-home-claude/999d0128-2648-5e0d-9375-1863dd874899/scratchpad/mine.csv', index=False)

# Comparisons
for setn, d in (('QC held-back', qc), ('MapBiomas lockbox', lb), ('Lockbox agree>=2', lb[lb.agree >= 2].reset_index(drop=True))):
    y = d.label.values; f = y == 0; g = d.block.values; bl = np.unique(g); idx = {b: np.where(g == b)[0] for b in bl}
    sb = S[(setn, 'Baseline')]; eb = sb[f] > L['Baseline']['FE5']
    for k in ('Lean 4', 'P4'):
        s = S[(setn, k)]; em = s[f] > L[k]['FE5']
        fixed = int((eb & ~em).sum()); new = int((em & ~eb).sum())
        p_less = binomtest(new, fixed + new, 0.5, alternative='less').pvalue if fixed + new else 1
        p_greater = binomtest(new, fixed + new, 0.5, alternative='greater').pvalue if fixed + new else 1
        rng = np.random.default_rng(7); diffs = []
        for _ in range(2000):
            ii = np.concatenate([idx[b] for b in rng.choice(bl, len(bl))])
            if len(set(y[ii])) == 2: diffs.append(roc_auc_score(y[ii], s[ii]) - roc_auc_score(y[ii], sb[ii]))
        diffs = np.array(diffs)
        print(f'{setn:18s} {k:7s} dAUC={roc_auc_score(y,s)-roc_auc_score(y,sb):.3f} CI {np.percentile(diffs,2.5):.3f}..{np.percentile(diffs,97.5):.3f} p(d<=0)={np.mean(diffs<=0):.4f} '
              f'nblocks={len(bl)} FE model={em.sum()} base={eb.sum()} fixed={fixed} new={new} p_McN(H1 fewer)={p_less:.4f} p(other dir)={p_greater:.4f}')

# Integrity
print('id overlap dev/lockbox', len(set(dev_raw.id) & set(lb_raw.id)), 'dev/qc-heldback', len(set(dev_raw.id) & set(qc.id)), 'lockbox/qc', len(set(lb_raw.id) & set(qc.id)))
print('block overlap dev/lockbox', len(set(dev_raw.block) & set(lb_raw.block)), 'lockbox blocks', lb_raw.block.nunique(), 'after filter', lb.block.nunique())
aeqc = set(AE[AE.set == 'qc'].id); print('AE qc ids', len(aeqc), 'in sealed qc', len(aeqc & set(qc.id)), 'AE qc not in sealed', len(aeqc - set(qc.id)), 'sealed qc not in AE', len(set(qc.id) - aeqc))
aemb = set(AE[AE.set == 'mb'].id); print('AE mb ids', len(aemb), 'dev', len(aemb & set(dev_raw.id)), 'lb', len(aemb & set(lb_raw.id)), 'other', len(aemb - set(dev_raw.id) - set(lb_raw.id)))
print('column sets equal dev vs lockbox', list(dev_raw.columns) == list(lb_raw.columns))
print('QC-sealed missing cols vs lockbox', sorted(set(lb_raw.columns) - set(qc.columns)), 'extra', sorted(set(qc.columns) - set(lb_raw.columns)))
print('cutoff dev', dev_raw.cutoff.value_counts().to_dict(), 'lb', lb_raw.cutoff.value_counts().to_dict(), 'qc', qc.cutoff.value_counts().to_dict())
print('season_start dev', dev_raw.season_start.value_counts().head(5).to_dict(), 'lb', lb_raw.season_start.value_counts().head(5).to_dict(), 'qc', qc.season_start.value_counts().head(8).to_dict())
print('qc clear_month', qc.clear_month.value_counts().sort_index().to_dict())
cols = FEAT['P4'] + ['nfw', 'chm_mean_r15', 'chm_stdDev_r15', 'A_su_NDVI', 'su_n_clear']
out = []
for c in dict.fromkeys(cols):
    for lab in (0, 1):
        a = dev[dev.label == lab][c]; b = lb[lb.label == lab][c]; q = qc[qc.label == lab][c]
        out.append(dict(feat=c, lab=lab, dev_med=a.median(), lb_med=b.median(), qc_med=q.median(), dev_nan=a.isna().mean(), lb_nan=b.isna().mean(), qc_nan=q.isna().mean(),
                        ks_p_dev_lb=ks_2samp(a.dropna(), b.dropna()).pvalue, dev_iqr=a.quantile(.75) - a.quantile(.25)))
print(pd.DataFrame(out).round(3).to_string())
