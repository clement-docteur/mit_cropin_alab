"""Phase 0 – re-run the 4 Nov texture/SAG comparisons and the baseline models under the
v4.1 rules. Uses only MapBiomas dev and QC dev tables. Label 1 = plantation, 0 = forest.
Forest error (FE) = share of forest called plantation (our dangerous error)."""
import pandas as pd, numpy as np, warnings
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')
rng = np.random.default_rng(2026); B = 500; FES = [0.01, 0.02, 0.05]

mb = pd.read_csv('mb_dev_table.csv'); qc = pd.read_csv('qc_dev_table.csv')
n0 = len(mb); mb = mb[mb.su_NDVI > 0.6].copy()
print(f'MB dev tree filter (summer NDVI > 0.6): {n0} -> {len(mb)}')
qc['block'] = np.floor(qc.lon/0.25).astype(int).astype(str)+'_'+np.floor(qc.lat/0.25).astype(int).astype(str)
train_mask = mb.agree >= 2            # decision 5 Nov: agree = 1 out of training, kept in evaluation

def rec_at(y, s, fe):
    t = np.quantile(s[y == 0], 1 - fe, method='higher'); return (s[y == 1] > t).mean(), t
def boot(y, s, g, fn):
    blocks = np.unique(g); idx = {b: np.where(g == b)[0] for b in blocks}; out = []
    for _ in range(B):
        ii = np.concatenate([idx[b] for b in rng.choice(blocks, len(blocks))])
        if len(np.unique(y[ii])) < 2: continue
        out.append(fn(y[ii], s[ii]))
    return np.percentile(out, [2.5, 97.5])
def summarize(y, s, g, name, fes=FES):
    y, s, g = np.asarray(y), np.asarray(s, float), np.asarray(g)
    auc = roc_auc_score(y, s); lo, hi = boot(y, s, g, roc_auc_score)
    row = dict(name=name, n_forest=int((y==0).sum()), n_plant=int((y==1).sum()), AUC=auc, AUC_lo=lo, AUC_hi=hi)
    for fe in fes:
        r, _ = rec_at(y, s, fe); rl, rh = boot(y, s, g, lambda a, b: rec_at(a, b, fe)[0])
        row[f'rec@{int(fe*100)}%'] = r; row[f'rec@{int(fe*100)}%_CI'] = f'{rl:.2f}–{rh:.2f}'
    return row

# ---------- 1. single features from the 4 Nov comparison ----------
FEATS = {'SAG': 'SAG (Oct–Jan)', 'su_B11': 'SWIR B11 mean', 'su_B8': 'NIR B8 mean', 'su_NDVI': 'NDVI mean',
         'B11_contrast_w7': 'B11 contrast w7', 'B8_idm_w7': 'NIR homogeneity w7', 'B8_contrast_w3': 'NIR contrast w3',
         'nfw': 'NFW 2020 probability', 'chm_mean_r15': 'Meta height mean (15 m)', 'chm_stdDev_r15': 'Meta height SD (15 m)'}
rows = []
for f, lab in FEATS.items():
    for setname, d in [('MapBiomas dev', mb[train_mask]), ('QC dev', qc)]:
        y, x = d.label.values, d[f].values
        a = roc_auc_score(y, x); sign = 1 if a >= 0.5 else -1
        p = mannwhitneyu(x[y==1], x[y==0]).pvalue
        r = summarize(y, sign*x, d.block.values, lab)
        r.update(set=setname, direction='higher = plantation' if sign == 1 else 'lower = plantation', MW_p=p)
        rows.append(r)
single = pd.DataFrame(rows)

# ---------- 2. models: blocked CV on MapBiomas dev, then trained on all MB dev → QC dev ----------
MODELS = {'Spectral means': ['su_B5', 'su_B8', 'su_B11', 'su_NDVI'],
          'Spectral + NIR texture': ['su_B5', 'su_B8', 'su_B11', 'su_NDVI', 'B8_idm_w7', 'B8_contrast_w3'],
          'SAG alone': ['SAG'],
          'SAG + spectral + NIR texture': ['SAG', 'su_B5', 'su_B8', 'su_B11', 'su_NDVI', 'B8_idm_w7', 'B8_contrast_w3'],
          'Baseline (MB): SAG + NFW + height + height SD': ['SAG', 'nfw', 'chm_mean_r15', 'chm_stdDev_r15']}
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
gkf = GroupKFold(n_splits=5); mrows = []; oof_all = {}
for name, cols in MODELS.items():
    oof = np.full(len(mb), np.nan)
    for tr, te in gkf.split(mb, mb.label, mb.block):
        trm = tr[train_mask.values[tr]]
        mdl = lr().fit(mb.iloc[trm][cols], mb.iloc[trm].label); oof[te] = mdl.predict_proba(mb.iloc[te][cols])[:, 1]
    oof_all[name] = oof
    r = summarize(mb.label, oof, mb.block, name); r['set'] = 'MapBiomas dev (blocked CV)'; mrows.append(r)
    full = lr().fit(mb[train_mask][cols], mb[train_mask].label); s = full.predict_proba(qc[cols])[:, 1]
    r = summarize(qc.label, s, qc.block, name, fes=[0.05]); r['set'] = 'QC dev (trained on MB)'
    t = rec_at(mb.label.values, oof, 0.05)[1]   # threshold fixed on MapBiomas at 5% FE, then applied to QC
    r['QC_recall_at_MB_threshold'] = (s[qc.label==1] > t).mean(); r['QC_FE_at_MB_threshold'] = (s[qc.label==0] > t).mean()
    mrows.append(r)
models = pd.DataFrame(mrows)

# ---------- 3. error groups for the best MB model (5% FE threshold on OOF scores) ----------
best = 'SAG + spectral + NIR texture'; s = oof_all[best]; t = rec_at(mb.label.values, s, 0.05)[1]
mb['flag'] = (s > t).astype(int); grows = []
for gname, mask in [('All', mb.index == mb.index), ('Interior (border=0)', mb.border == 0), ('Border (border=1)', mb.border == 1),
                    ('Agree 3/3', mb.agree == 3), ('Agree 2/3', mb.agree == 2), ('Agree 1/3 (not trained on)', mb.agree == 1),
                    ('Atlantic Forest', mb.biome.str.startswith('MATA')), ('Pampa', mb.biome == 'PAMPA'),
                    ('PR', mb.state == 'PR'), ('SC', mb.state == 'SC'), ('RS', mb.state == 'RS')]:
    d = mb[mask]
    grows.append(dict(group=gname, n_forest=int((d.label==0).sum()), n_plant=int((d.label==1).sum()),
                      forest_error=d[d.label==0].flag.mean(), plant_recall=d[d.label==1].flag.mean()))
groups = pd.DataFrame(grows)
for name, df in [('single', single), ('models', models), ('groups', groups)]:
    df.to_csv(f'p0_{name}.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(single[['set','name','n_forest','n_plant','direction','AUC','AUC_lo','AUC_hi','rec@1%','rec@5%','rec@5%_CI']].round(3).to_string())
print(models.round(3).to_string()); print(groups.round(3).to_string())
