"""Phase 5 – robustness of the Phase 4 candidates (decision 6 Nov: Lean 4 = main, P4 10-feature = challenger).
Tests: leave-one-state-out, leave-one-biome-out, clear-image terciles, single-year features (year to year).
Thresholds are always set on training data only (5% forest error on training out-of-fold scores)."""
import numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import p5_common as c
mb, qc = c.load(); y = mb.label.values; trm = (mb.agree >= 2).values
dist = ((mb.label == 0) & (mb.D_n_events > 0)).values
CAND = {'Baseline': c.BASE,
        'Lean 4': ['B_SAG_phase_sin', 'A_su_NDMI', 'B_SAG_SON', 'A_sg_NDVI'],
        'P4 main (10)': ['B_SAG_phase_sin', 'A_su_NDMI', 'B_NDRE_phase_sin', 'B_SAG_SON', 'A_sg_NDVI', 'B_SWIR_phase_sin', 'B_SAG_mean', 'B_SWIR_phase_cos', 'B_SAG_DJ', 'A_SAG']}
def wilson(k, n):
    if n == 0: return (np.nan, np.nan)
    p = k / n; z = 1.96; d = 1 + z * z / n; m = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (m - h, m + h)
def fit_thr(d, cols):
    """Train on d (agree>=2), threshold = 5% FE on d's own blocked out-of-fold scores."""
    t_ = d[d.agree >= 2]; s = np.full(len(t_), np.nan)
    for a, b in GroupKFold(5).split(t_, t_.label, t_.block):
        s[b] = c.lr().fit(t_.iloc[a][cols], t_.iloc[a].label).predict_proba(t_.iloc[b][cols])[:, 1]
    return c.lr().fit(t_[cols], t_.label), c.thr(t_.label.values, s, 0.05)
def score(m, t, d, cols, name, group):
    s = m.predict_proba(d[cols])[:, 1]; yy = d.label.values; k = int((s[yy == 0] > t).sum()); n = int((yy == 0).sum())
    lo, hi = wilson(k, n)
    return dict(model=name, test=group, n_forest=n, n_plant=int((yy == 1).sum()), AUC=roc_auc_score(yy, s) if len(set(yy)) == 2 else np.nan,
                FE=k / n, FE_CI=f'{lo:.2f}–{hi:.2f}', recall=(s[yy == 1] > t).mean())
rows = []
# 1–2. leave one state / biome out
mb['biome2'] = np.where(mb.biome.str.startswith('MATA'), 'Atlantic Forest', 'Pampa')
for name, cols in CAND.items():
    for key, col in [('state', 'state'), ('biome', 'biome2')]:
        for g in sorted(mb[col].unique()):
            m, t = fit_thr(mb[mb[col] != g], cols); rows.append(score(m, t, mb[mb[col] == g], cols, name, f'leave-one-{key}-out: {g}'))
    print('done', name, file=sys.stderr)
# 3. clear-image terciles (standard blocked OOF, overall threshold)
mb['ter'] = pd.qcut(mb.su_n_clear, 3, labels=['few (1–9)', 'mid (10–16)', 'many (17+)'])
for name, cols in CAND.items():
    s = c.oof(mb, cols, trm); t = c.thr(y, s, 0.05)
    for g in mb.ter.cat.categories:
        k = (mb.ter == g).values; yy = y[k]
        rows.append(dict(model=name, test=f'clear images: {g}', n_forest=int((yy == 0).sum()), n_plant=int((yy == 1).sum()),
                         AUC=roc_auc_score(yy, s[k]), FE=(s[k][yy == 0] > t).mean(), recall=(s[k][yy == 1] > t).mean()))
    rows.append(dict(model=name, test='disturbed natural forest (n=49)', n_forest=int(dist.sum()), FE=(s[dist] > t).mean()))
R = pd.DataFrame(rows)
# 4. year to year (monthly-based features)
F = pd.read_csv('p5_year_features.csv')
MON = {'Lean 4 (monthly-based)': ['B_SAG_phase_sin', 'M_su_NDMI', 'B_SAG_SON', 'M_sg_NDVI'],
       'P4 main (monthly-based)': ['B_SAG_phase_sin', 'M_su_NDMI', 'B_NDRE_phase_sin', 'B_SAG_SON', 'M_sg_NDVI', 'B_SWIR_phase_sin', 'B_SAG_mean', 'B_SWIR_phase_cos', 'B_SAG_DJ', 'M_SAG_octjan']}
key = mb[['id', 'label', 'block', 'agree']].assign(set='mb')
def win(set_, w, base): return base.merge(F[(F.set == set_) & (F.window == w)], on=['id', 'set'], how='left')
yrows = []
for name, cols in MON.items():
    M3 = win('mb', '3yr', key)
    # (a) frozen 3-year model, scored on single years (out-of-fold for MapBiomas points)
    s3 = np.full(len(M3), np.nan); tm = (M3.agree >= 2).values; sY = {yr: np.full(len(M3), np.nan) for yr in ('2020', '2021', '2022')}
    Ys = {yr: win('mb', yr, key) for yr in sY}
    for a, b in GroupKFold(5).split(M3, M3.label, M3.block):
        a = a[tm[a]]; m = c.lr().fit(M3.iloc[a][cols], M3.iloc[a].label)
        s3[b] = m.predict_proba(M3.iloc[b][cols])[:, 1]
        for yr in sY: sY[yr][b] = m.predict_proba(Ys[yr].iloc[b][cols])[:, 1]
    t = c.thr(y, s3, 0.05)
    yrows.append(dict(model=name, train='MB 3-yr', test='MB 3-yr (sanity)', AUC=roc_auc_score(y, s3), FE=(s3[y == 0] > t).mean(), recall=(s3[y == 1] > t).mean()))
    for yr in sY:
        s = sY[yr]; yrows.append(dict(model=name, train='MB 3-yr', test=f'MB {yr} only', AUC=roc_auc_score(y, s), FE=(s[y == 0] > t).mean(), recall=(s[y == 1] > t).mean()))
    full = c.lr().fit(M3[tm][cols], M3[tm].label)
    qk = qc[['id', 'label']].assign(set='qc')
    for w in ('3yr', '2023', '2024', '2025'):
        Q = win('qc', w, qk); ok = Q[cols].notna().any(axis=1).values; s = full.predict_proba(Q[ok][cols])[:, 1]; yy = Q[ok].label.values
        k = int((s[yy == 0] > t).sum()); n = int((yy == 0).sum()); lo, hi = wilson(k, n)
        yrows.append(dict(model=name, train='MB 3-yr', test=f'QC {w}', n_forest=n, AUC=roc_auc_score(yy, s), FE=k / n, FE_CI=f'{lo:.2f}–{hi:.2f}', recall=(s[yy == 1] > t).mean()))
    # (b) cross-year: train on year A (training blocks), test on year B (held-out blocks)
    for ya in ('2020', '2021', '2022'):
        for yb in ('2020', '2021', '2022'):
            A, Bd = Ys[ya], Ys[yb]; s = np.full(len(A), np.nan); sa = np.full(len(A), np.nan)
            for a, b in GroupKFold(5).split(A, A.label, A.block):
                a = a[tm[a]]; m = c.lr().fit(A.iloc[a][cols], A.iloc[a].label)
                s[b] = m.predict_proba(Bd.iloc[b][cols])[:, 1]; sa[b] = m.predict_proba(A.iloc[b][cols])[:, 1]
            t2 = c.thr(y, sa, 0.05)
            yrows.append(dict(model=name, train=f'MB {ya}', test=f'MB {yb}', AUC=roc_auc_score(y, s), FE=(s[y == 0] > t2).mean(), recall=(s[y == 1] > t2).mean()))
    print('done', name, file=sys.stderr)
Y = pd.DataFrame(yrows)
rain = pd.read_csv('gee/w2_chirps.csv'); rain['yr'] = rain.month.str[:4]
RR = rain.groupby(['set', 'yr']).apply(lambda g: g.rain_mm.sum() / g.clim_mm.sum()).round(2).rename('rain_vs_normal').reset_index()
R.to_csv('p5_robust.csv', index=False); Y.to_csv('p5_years.csv', index=False); RR.to_csv('p5_rain_by_year.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(R.round(3).to_string()); print(Y.round(3).to_string()); print(RR[RR.yr.astype(int) >= 2020].to_string())
