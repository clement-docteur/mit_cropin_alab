"""Shared helpers for Wave 1 analysis (metrics, bootstrap, models, alert-like weights)."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')
RNG = np.random.default_rng(2026)
FES = [0.01, 0.02, 0.05]
BASE = ['A_SAG', 'chm_mean_r15', 'chm_stdDev_r15']          # decision-rule baseline (NFW-free), §10
BASE_NFW = ['A_SAG', 'nfw', 'chm_mean_r15', 'chm_stdDev_r15']
CTX = ['ctx_log_patch', 'ctx_log_dist', 'wc_tree_frac_50m', 'wc_tree_frac_100m', 'wc_tree_frac_250m', 'slope']

def load():
    mb = pd.read_csv('w2_mb_dev_features.csv'); qc = pd.read_csv('w2_qc_dev_features.csv')
    mb = mb[mb.A_su_NDVI > 0.6].reset_index(drop=True)      # tree filter (Phase 0)
    for d in (mb, qc):
        d['ctx_log_patch'] = np.log1p(d.wc_patch_px); d['ctx_log_dist'] = np.log1p(d.wc_dist_nontree_m)
    return mb, qc

def wquant(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) / w.sum()
    return x[min(np.searchsorted(c, q), len(x) - 1)]
def thr(y, s, fe, w=None):
    f = s[y == 0]
    return np.quantile(f, 1 - fe, method='higher') if w is None else wquant(f, w[y == 0], 1 - fe)
def rec_fe(y, s, t, w=None):
    w = np.ones(len(y)) if w is None else w
    p, f = y == 1, y == 0
    return np.average(s[p] > t, weights=w[p]), np.average(s[f] > t, weights=w[f])
def rec_at(y, s, fe): return rec_fe(y, s, thr(y, s, fe))[0]
def bci(y, s, g, fn, B=300):
    y, s, g = np.asarray(y), np.asarray(s, float), np.asarray(g)
    bl = np.unique(g); idx = {b: np.where(g == b)[0] for b in bl}; out = []
    for _ in range(B):
        ii = np.concatenate([idx[b] for b in RNG.choice(bl, len(bl))])
        if len(np.unique(y[ii])) == 2: out.append(fn(y[ii], s[ii]))
    return np.percentile(out, [2.5, 97.5])
def lr(C=1.0, penalty='l2'):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=C, penalty=penalty, solver='liblinear' if penalty == 'l1' else 'lbfgs', max_iter=3000))
def oof(mb, cols, train_mask):
    s = np.full(len(mb), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(mb, mb.label, mb.block):
        trm = tr[train_mask[tr]]
        s[te] = lr().fit(mb.iloc[trm][cols], mb.iloc[trm].label).predict_proba(mb.iloc[te][cols])[:, 1]
    return s

def alert_weights(mb, qc):
    """Within each class: P(looks like QC dev | context). w = p/(1-p) * nMB/nQC, clipped at 99th pct, mean 1."""
    mb = mb.copy(); mb['alert_p'] = np.nan; mb['alert_w'] = np.nan; info = {}
    for c in (0, 1):
        a, b = mb[mb.label == c], qc[qc.label == c]
        X = pd.concat([a[CTX], b[CTX]]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
        m = lr().fit(X, y); p = m.predict_proba(a[CTX])[:, 1]
        w = p / (1 - p) * len(a) / len(b); w = np.minimum(w, np.percentile(w, 99)); w = w / w.mean()
        mb.loc[a.index, 'alert_p'] = p; mb.loc[a.index, 'alert_w'] = w
        info[c] = dict(n=len(a), n_eff=w.sum() ** 2 / (w ** 2).sum(), auc_qc_vs_mb=roc_auc_score(y, m.predict_proba(X)[:, 1]))
    mb['alert_top25'] = mb.groupby('label').alert_p.transform(lambda s: s >= s.quantile(0.75))
    return mb, info

def evaluate(mb, qc, cols, name, B=300):
    """Full report for one model: MB blocked CV + alert-like calibration + QC dev transfer."""
    tm = (mb.agree >= 2).values; y = mb.label.values; g = mb.block.values
    s = oof(mb, cols, tm); r = dict(model=name, n_features=len(cols), features='; '.join(cols))
    r['MB_AUC'] = roc_auc_score(y, s); r['MB_AUC_lo'], r['MB_AUC_hi'] = bci(y, s, g, roc_auc_score, B)
    for fe in FES:
        k = f'{int(fe*100)}%'; r[f'MB_rec@{k}'] = rec_at(y, s, fe)
        lo, hi = bci(y, s, g, lambda a, b: rec_at(a, b, fe), B); r[f'MB_rec@{k}_CI'] = f'{lo:.2f}–{hi:.2f}'
    t_u = thr(y, s, 0.05); w = mb.alert_w.values; t_w = thr(y, s, 0.05, w)
    r['MB_weighted_rec@5%'], _ = rec_fe(y, s, t_w, w)
    top = mb.alert_top25.values
    r['MBtop25_rec_at_unweighted_t'], r['MBtop25_FE_at_unweighted_t'] = rec_fe(y[top], s[top], t_u)
    r['MBtop25_rec_at_weighted_t'], r['MBtop25_FE_at_weighted_t'] = rec_fe(y[top], s[top], t_w)
    a1 = mb.agree.values == 1
    r['MB_agree1_FE_at_t'] = (s[a1 & (y == 0)] > t_u).mean(); r['MB_agree1_rec_at_t'] = (s[a1 & (y == 1)] > t_u).mean()
    full = lr().fit(mb[tm][cols], y[tm]); q = full.predict_proba(qc[cols])[:, 1]; yq = qc.label.values
    r['QC_AUC'] = roc_auc_score(yq, q); r['QC_AUC_lo'], r['QC_AUC_hi'] = bci(yq, q, qc.block.values, roc_auc_score, B)
    r['QC_rec_at_MB_t'], r['QC_FE_at_MB_t'] = rec_fe(yq, q, t_u)
    r['QC_rec_at_weighted_t'], r['QC_FE_at_weighted_t'] = rec_fe(yq, q, t_w)
    r['QC_rec@5%_own'] = rec_at(yq, q, 0.05)
    grp = {}
    for gname, m in [('interior', mb.border == 0), ('border', mb.border == 1), ('agree3', mb.agree == 3),
                     ('agree2', mb.agree == 2), ('PR', mb.state == 'PR'), ('SC', mb.state == 'SC'), ('RS', mb.state == 'RS'),
                     ('Atlantic', mb.biome.str.startswith('MATA')), ('Pampa', mb.biome == 'PAMPA')]:
        m = m.values; grp[gname] = rec_fe(y[m], s[m], t_u)
    for k, (rc, fe) in grp.items(): r[f'grp_{k}_FE'] = fe; r[f'grp_{k}_rec'] = rc
    return r, s
