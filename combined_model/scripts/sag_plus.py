"""'Cropin's SAG + a few features' (development half only; held-back already used).
Rule fixed before running: start from Cropin's SAG (nc_A_SAG, one season Oct–Jan). Forward selection, logistic regression
(L2, C = 1, standardized), each step adds the feature that most raises MapBiomas blocked-CV AUC (5 folds by block, training
points only). Stop after 5 additions or when the gain is < 0.005. QC dev is only for reporting. Run twice: all features,
and without MapBiomas-derived features. Nested check: selection repeated inside each outer fold."""
import numpy as np, pandas as pd
exec(open('search_v2.py').read().split("rows, oofs, qcs = [], {}, {}")[0])
MBD = ['nc_H_mapbiomas10m_plantation', 'lh_natural_run_years', 'lh_n_changes', 'lh_ever_open']
def cv_scores(cols, TT, yy, gg):
    s = np.full(len(TT), np.nan)
    for tr, te in GroupKFold(5).split(TT, yy, gg):
        s[te] = lr(1).fit(TT.iloc[tr][cols], yy[tr]).predict_proba(TT.iloc[te][cols])[:, 1]
    return s
def forward(pool, TT, yy, gg):
    cols = ['nc_A_SAG']; path = [(list(cols), roc_auc_score(yy, cv_scores(cols, TT, yy, gg)))]
    for _ in range(5):
        best = max(((roc_auc_score(yy, cv_scores(cols + [c], TT, yy, gg)), c) for c in pool if c not in cols))
        if best[0] - path[-1][1] < 0.005: break
        cols = cols + [best[1]]; path.append((list(cols), best[0]))
    return path
rows = []
for variant, pool in (('all features', P), ('without MapBiomas-derived', [c for c in P if c not in MBD])):
    path = forward(pool, T, y, g)
    for cols, a in path:
        so = cv_scores(cols, T, y, g); t = thr(y, so)
        m = lr(1).fit(T[cols], y); sq = m.predict_proba(Q[cols])[:, 1]
        f = np.sort(sq[yq == 0])[::-1]
        rows.append({'variant': variant, 'step': len(cols) - 1, 'added': cols[-1] if len(cols) > 1 else 'Cropin SAG alone',
                     'MB CV AUC': round(a, 3), 'MB recall @5%': round(float((so[y == 1] > t).mean()), 3),
                     'QC AUC': round(roc_auc_score(yq, sq), 3), 'QC forest errors': f'{int((sq[yq == 0] > t).sum())}/58',
                     'QC recall': round(float((sq[yq == 1] > t).mean()), 3),
                     'QC recall @3/58 errors': round(float((sq[yq == 1] > f[3]).mean()), 3)})
        print(rows[-1], flush=True)
    final = path[-1][0]; coef = lr(1).fit(T[final], y)[-1].coef_[0]
    print(variant, 'standardized coefficients:', dict(zip(final, np.round(coef, 2))))
    # nested check: redo selection inside each outer fold
    s = np.full(len(T), np.nan); picks = []
    for tr, te in GroupKFold(5).split(T, y, g):
        Ti = T.iloc[tr].reset_index(drop=True); c = forward(pool, Ti, y[tr], g[tr])[-1][0]; picks.append(c[1:])
        s[te] = lr(1).fit(Ti[c], y[tr]).predict_proba(T.iloc[te][c])[:, 1]
    print(variant, 'nested MB AUC:', round(roc_auc_score(y, s), 3), 'picks per fold:', picks, flush=True)
pd.DataFrame(rows).to_csv('sag_plus_results.csv', index=False); pd.set_option('display.width', 220); print(pd.DataFrame(rows).to_string(index=False))
