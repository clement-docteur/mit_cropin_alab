"""Wave 1 – nested blocked CV (reviewer item 3): feature screening (criteria 1–3, no QC data)
and stability selection are redone inside each outer fold, so MapBiomas scores are not optimistic."""
import numpy as np, pandas as pd, re, sys
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import w2_common as c
src = open('w2_screen.py').read(); exec(src[src.index('def prior'):src.index('fams = ')])   # same priors as the screen
mb, qc = c.load(); y = mb.label.values
fams = {p: [x for x in mb.columns if x.startswith(p + '_') and x not in ('B_n_months', 'B_n_drought_months')] for p in 'BF'}
rng = np.random.default_rng(7)
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out
def select(tr, fam):
    rows = []
    for f in fams[fam]:
        d = tr[tr[f].notna()]; x = d[f].values; yy = d.label.values
        a = roc_auc_score(yy, x); sg = 1 if a >= .5 else -1
        lo = c.bci(yy, sg * x, d.block.values, roc_auc_score, 100)[0]
        st = [roc_auc_score(g.label, g[f]) for _, g in d.groupby('state') if g.label.nunique() == 2]
        pr = prior(f)
        rows.append((f, mannwhitneyu(x[yy == 1], x[yy == 0]).pvalue, lo, all((v - .5) * sg > 0 for v in st) and (pr in (0, sg)), max(a, 1 - a)))
    R = pd.DataFrame(rows, columns=['f', 'p', 'lo', 'dir', 'auc']); R['q'] = bh(R.p)
    cand = R[(R.q < .05) & (R.lo > .60) & R.dir].sort_values('auc', ascending=False).f.tolist()
    if not cand: return []
    C = tr[cand].corr(method='spearman').abs(); keep = []
    for f in cand:
        if all(C.loc[f, k] <= .9 for k in keep): keep.append(f)
    bl = tr.block.unique(); cnt = pd.Series(0., index=keep)
    for _ in range(50):
        sub = tr[tr.block.isin(rng.choice(bl, len(bl) // 2, replace=False))]
        m = c.lr(C=.05, penalty='l1').fit(sub[keep], sub.label); cnt += (np.abs(m[-1].coef_[0]) > 1e-6)
    fr = (cnt / 50).sort_values(ascending=False); return list(fr[fr >= .6].index[:10]) or list(fr.index[:5])
oofs = {k: np.full(len(mb), np.nan) for k in ['Baseline + B', 'Baseline + F', 'Baseline + B + F']}
chosen = []
for k, (tr_i, te_i) in enumerate(GroupKFold(5).split(mb, y, mb.block)):
    tr = mb.iloc[tr_i]; tr = tr[tr.agree >= 2]; sel = {f: select(tr, f) for f in 'BF'}
    chosen.append({f: sel[f] for f in sel}); print('fold', k, sel, file=sys.stderr)
    for name, fs in [('Baseline + B', sel['B']), ('Baseline + F', sel['F']), ('Baseline + B + F', sel['B'] + sel['F'])]:
        cols = list(dict.fromkeys(c.BASE + fs))
        oofs[name][te_i] = c.lr().fit(tr[cols], tr.label).predict_proba(mb.iloc[te_i][cols])[:, 1]
rows = []
for name, s in oofs.items():
    r = dict(model=name + ' (nested CV)', MB_AUC=roc_auc_score(y, s))
    r['MB_AUC_lo'], r['MB_AUC_hi'] = c.bci(y, s, mb.block.values, roc_auc_score, 300)
    for fe in c.FES: r[f'MB_rec@{int(fe*100)}%'] = c.rec_at(y, s, fe)
    rows.append(r)
N = pd.DataFrame(rows); N.to_csv('w2_nested.csv', index=False)
pd.DataFrame(chosen).to_csv('w2_nested_selected.csv', index=False)
print(N.round(3).to_string())
