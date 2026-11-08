"""Phase 4 – parsimony check: do fewer, less collinear features do as well as the 10-feature model?"""
import numpy as np, pandas as pd, sys
import p4_common as c
mb, qc = c.load(); mb, _ = c.alert_weights(mb, qc); y = mb.label.values
dist = ((mb.label == 0) & (mb.D_n_events > 0)).values
V = {'Lean 5: SAG phase, NDMI, NDRE phase, SAG Sep–Nov, NDVI Oct–Jan': ['B_SAG_phase_sin', 'A_su_NDMI', 'B_NDRE_phase_sin', 'B_SAG_SON', 'A_sg_NDVI'],
     'Lean 4: SAG phase, NDMI, SAG Sep–Nov, NDVI Oct–Jan': ['B_SAG_phase_sin', 'A_su_NDMI', 'B_SAG_SON', 'A_sg_NDVI'],
     'Lean 3: SAG Sep–Nov, SAG phase, NDMI': ['B_SAG_SON', 'B_SAG_phase_sin', 'A_su_NDMI'],
     'Single: SAG Sep–Nov': ['B_SAG_SON'], 'Single: SAG Dec–Jan (Cropin window, monthly)': ['B_SAG_DJ']}
rows = []
for k, cols in V.items():
    r, s = c.evaluate(mb, qc, cols, k); t = c.thr(y, s, 0.05); r['FE_disturbed_forest'] = (s[dist] > t).mean()
    r['coefs'] = str(dict(zip(cols, c.lr().fit(mb[mb.agree >= 2][cols], mb[mb.agree >= 2].label)[-1].coef_[0].round(2))))
    rows.append(r); print('done', k, file=sys.stderr)
R = pd.DataFrame(rows); R.to_csv('p4_lean.csv', index=False); pd.set_option('display.width', 320)
print(R[['model', 'MB_AUC', 'MB_rec@1%', 'MB_rec@2%', 'MB_rec@5%', 'QC_AUC', 'QC_AUC_lo', 'QC_AUC_hi', 'QC_FE_at_MB_t', 'QC_rec_at_MB_t', 'QC_FE_at_weighted_t', 'QC_rec_at_weighted_t', 'FE_disturbed_forest', 'grp_border_FE']].round(3).to_string())
for r in rows: print(r['model'][:12], r['coefs'])
