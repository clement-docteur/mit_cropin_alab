"""Combined model search v1 (preliminary; protocol PROTOCOL_search_v1.md). Development data only."""
import numpy as np, pandas as pd, json, warnings, sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
warnings.filterwarnings('ignore')
RNG = np.random.default_rng(2026)
M = pd.read_csv('/home/claude/master/out/master_features_v2.csv')
M = M.merge(pd.read_csv('/home/claude/combined/rh_features.csv'), on=['id', 'set'], how='left')
W = pd.concat([pd.read_csv(f) for f in ['/home/claude/w3/w3_mb_dev_features.csv', '/home/claude/p7/sealed/w3_mb_dev_features.csv',
               '/home/claude/w3/w3_qc_dev_features.csv', '/home/claude/p7/sealed/w3_qc_dev_features.csv']])[['id', 'set', 'H_hansen_tc2000']]
M = M.merge(W.rename(columns={'H_hansen_tc2000': 'nc_hansen_tc2000'}), on=['id', 'set'], how='left')
A = pd.read_csv('/home/claude/master/out/master_alphaearth_v1.csv'); AE = [f'ae_A{i:02d}' for i in range(64)]
M = M.merge(A[['id', 'set'] + AE], on=['id', 'set'], how='left')
dec = pd.read_csv('/home/claude/master/out/feature_decisions_v2.csv')
P = [c for c in dec[dec.role == 'feature'].column if not c.startswith('hc_') and c != 'mm_B6_win']
P += ['rh_n_cycles', 'rh_s_std', 'rh_s_frac_below_forest', 'rh_abrupt_start', 'nc_hansen_tc2000']
T = M[M.train_eligible].reset_index(drop=True); Q = M[(M.set == 'qc') & (M.split == 'dev')].reset_index(drop=True)
rho = T[['rh_max_drop', 'rh_n_cycles']].corr('spearman').iloc[0, 1]
if abs(rho) < 0.85: P.append('rh_max_drop')
PBB = P + AE
y, g = T.label.values, T.block.values; yq = Q.label.values
log = {'P_int': P, 'rho_maxdrop_ncycles': float(rho), 'n_train': int(len(T)), 'n_qc_dev': int(len(Q))}

def lr(C, pen='l2'): return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                          LogisticRegression(C=C, penalty=pen, solver='liblinear' if pen == 'l1' else 'lbfgs', max_iter=5000))
def gam(C): return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), SplineTransformer(n_knots=4, degree=3),
                                 LogisticRegression(C=C, max_iter=5000))
def tree(d): return make_pipeline(SimpleImputer(strategy='median'), DecisionTreeClassifier(max_depth=d, min_samples_leaf=20, random_state=0))
def rf(l): return make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(500, min_samples_leaf=l, n_jobs=-1, random_state=0))
def hgb(d): return HistGradientBoostingClassifier(max_depth=d, learning_rate=0.05, max_iter=300, early_stopping=True, random_state=0)
def mlp(a): return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                 MLPClassifier((32, 16), alpha=a, early_stopping=True, max_iter=2000, random_state=0))

def tuned(factory, grid, X, yy, gg):
    """pick the grid value with best inner grouped-CV AUC, return fitted model and choice"""
    if len(grid) == 1: return factory(grid[0]).fit(X, yy), grid[0]
    best = None
    for v in grid:
        s = np.full(len(yy), np.nan)
        for tr, te in GroupKFold(4).split(X, yy, gg):
            s[te] = factory(v).fit(X.iloc[tr], yy[tr]).predict_proba(X.iloc[te])[:, 1]
        a = roc_auc_score(yy, s)
        if best is None or a > best[0]: best = (a, v)
    return factory(best[1]).fit(X, yy), best[1]

def l1_selected(X, yy, gg):
    m, C = tuned(lambda c: lr(c, 'l1'), [0.02, 0.05, 0.1, 0.3], X, yy, gg)
    coef = m[-1].coef_[0]; return [c for c, w in zip(X.columns, coef) if w != 0] or list(X.columns[:1])

class GAMonL1:
    """I3: additive splines on the features an L1 logistic keeps"""
    def __init__(self, C): self.C = C
    def fit(self, X, yy, gg=None):
        self.cols = l1_selected(X, yy, gg if gg is not None else np.arange(len(yy)) % 4)
        self.m = gam(self.C).fit(X[self.cols], yy); return self
    def predict_proba(self, X): return self.m.predict_proba(X[self.cols])

MODELS = {  # id: (pool, kind, factory, grid)
 'I1': (P, 'int', lambda v: lr(v), [0.1, 1]), 'I2': (P, 'int', lambda v: lr(v, 'l1'), [0.02, 0.05, 0.1, 0.3]),
 'I3': (P, 'int', None, [0.1, 1]), 'I4': (P, 'int', tree, [3, 4]),
 'B1': (P, 'bb', rf, [5, 20]), 'B2': (P, 'bb', hgb, [2, 3]), 'B3': (P, 'bb', mlp, [1e-3, 1e-2]),
 'B4': (AE, 'bb', lambda v: lr(v), [0.1, 1]),
 'B5': (PBB, 'bb', rf, [5, 20]), 'B6': (PBB, 'bb', hgb, [2, 3]), 'B7': (PBB, 'bb', mlp, [1e-3, 1e-2])}

def fit_model(mid, X, yy, gg):
    pool, kind, fac, grid = MODELS[mid]
    if mid == 'I3':
        best = None
        for C in grid:
            s = np.full(len(yy), np.nan)
            for tr, te in GroupKFold(4).split(X, yy, gg):
                s[te] = GAMonL1(C).fit(X.iloc[tr], yy[tr], gg[tr]).predict_proba(X.iloc[te])[:, 1]
            a = roc_auc_score(yy, s); best = (a, C) if best is None or a > best[0] else best
        return GAMonL1(best[1]).fit(X, yy, gg), best[1]
    return tuned(fac, grid, X, yy, gg)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f'{max(0, m - h):.2f}–{min(1, m + h):.2f}'
def boot_auc(yy, s, gg, B=1000):
    bl = np.unique(gg); idx = {b: np.where(gg == b)[0] for b in bl}; out = []
    for _ in range(B):
        ii = np.concatenate([idx[b] for b in RNG.choice(bl, len(bl))])
        if len(set(yy[ii])) == 2: out.append(roc_auc_score(yy[ii], s[ii]))
    return np.percentile(out, [2.5, 97.5])
def thr(yy, s, fe=0.05): return np.quantile(s[yy == 0], 1 - fe, method='higher')

edge = (T.nc_wc_dist_nontree_m <= 10).values
rows, oofs, qcs = [], {}, {}
for mid in ['REF'] + list(MODELS):
    if mid == 'REF':
        s_oof = T.nc_A_SAG.values.astype(float); s_q = Q.nc_A_SAG.values.astype(float); choice = '-'; feats = ['nc_A_SAG']
    else:
        pool = MODELS[mid][0]; X = T[pool]; s_oof = np.full(len(T), np.nan)
        for tr, te in GroupKFold(5).split(X, y, g):
            m, _ = fit_model(mid, X.iloc[tr], y[tr], g[tr]); s_oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        mfull, choice = fit_model(mid, X, y, g); s_q = mfull.predict_proba(Q[pool])[:, 1]
        feats = mfull.cols if mid == 'I3' else (pool if mid != 'I2' else [c for c, w in zip(pool, mfull[-1].coef_[0]) if w != 0])
        if mid == 'I4': feats = sorted({pool[i] for i in mfull[-1].tree_.feature if i >= 0})
    t = thr(y, s_oof); oofs[mid], qcs[mid] = s_oof, s_q
    fq = int((s_q[yq == 0] > t).sum()); nf = int((yq == 0).sum())
    lo, hi = boot_auc(y, s_oof, g); qlo, qhi = boot_auc(yq, s_q, Q.block.values)
    rows.append({'model': mid, 'choice': choice, 'n_features': len(feats),
                 'MB OOF AUC': f'{roc_auc_score(y, s_oof):.3f} [{lo:.2f}–{hi:.2f}]', 'MB recall @5%': round(float((s_oof[y == 1] > t).mean()), 3),
                 'MB edge FE (≤10 m)': f'{int((s_oof[(y == 0) & edge] > t).sum())}/{int(((y == 0) & edge).sum())}',
                 'QC AUC': f'{roc_auc_score(yq, s_q):.3f} [{qlo:.2f}–{qhi:.2f}]',
                 'QC forest errors': f'{fq}/{nf} ({wilson(fq, nf)})', 'QC recall': round(float((s_q[yq == 1] > t).mean()), 3),
                 'qc_fe_n': fq, 'qc_recall_raw': float((s_q[yq == 1] > t).mean())})
    log[mid] = {'choice': str(choice), 'features': feats}
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv('/home/claude/combined/search_v1_results.csv', index=False)
pd.DataFrame(oofs).to_csv('/home/claude/combined/search_v1_oof.csv', index=False); pd.DataFrame(qcs).assign(uid=Q.uid).to_csv('/home/claude/combined/search_v1_qc.csv', index=False)
json.dump(log, open('/home/claude/combined/search_v1_log.json', 'w'), indent=1, default=str)
print(R.drop(columns=['qc_fe_n', 'qc_recall_raw']).to_string(index=False))
