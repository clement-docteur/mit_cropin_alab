"""Wave 1 – within-family models via stability selection (L1 logistic, 100 block subsamples),
then full evaluation (MB blocked CV, alert-like calibration, QC dev transfer)."""
import numpy as np, pandas as pd, sys
import w3_common as c
mb, qc = c.load(); mb, info = c.alert_weights(mb, qc)
S = pd.read_csv('w3_screen.csv'); trm = (mb.agree >= 2).values; tr = mb[trm]
rng = np.random.default_rng(2026)
def decorrelate(feats):
    """Drop a feature if |r| > 0.9 with a feature that has a higher screen AUC (keeps the model readable)."""
    feats = sorted(feats, key=lambda f: -S.set_index('feature').AUC[f]); keep = []
    C = tr[feats].corr(method='spearman').abs()
    for f in feats:
        if all(C.loc[f, k] <= 0.9 for k in keep): keep.append(f)
    return keep
def stability(feats, runs=100, C=0.05, thr=0.6, cap=10):
    bl = tr.block.unique(); cnt = pd.Series(0.0, index=feats)
    for _ in range(runs):
        sub = tr[tr.block.isin(rng.choice(bl, len(bl) // 2, replace=False))]
        m = c.lr(C=C, penalty='l1').fit(sub[feats], sub.label)
        cnt += (np.abs(m[-1].coef_[0]) > 1e-6).astype(float)
    freq = (cnt / runs).sort_values(ascending=False)
    sel = list(freq[freq >= thr].index[:cap]) or list(freq.index[:5])
    return sel, freq
out, sel_log, scores = [], {}, {}
def run(name, cols):
    r, s = c.evaluate(mb, qc, cols, name); out.append(r); scores[name] = s; print('done', name, file=sys.stderr)
run('Baseline (decision rule): SAG + height + height SD', c.BASE)
run('Comparison: SAG + NFW + height + height SD', c.BASE_NFW)
fam_sel = {}
for fam in 'EGH':
    for variant, mask in [('main', S.c1_q & S.c2_auc & S.c3_direction & ~S.context_dependent),
                          ('incl. context-dependent', S.c1_q & S.c2_auc & S.c3_direction)]:
        cand = S[(S.family == fam) & mask].feature.tolist()
        if len(cand) == 0: sel_log[f'{fam} {variant}'] = 'no candidates'; continue
        cand = decorrelate(cand); sel, freq = stability(cand)
        sel_log[f'{fam} {variant}'] = dict(n_candidates_after_decorrelation=len(cand), selected=sel,
                                           freq={k: round(v, 2) for k, v in freq.items() if v > 0})
        if variant == 'main': fam_sel[fam] = sel
        run(f'Family {fam} ({variant}) alone', sel)
        run(f'Baseline + Family {fam} ({variant})', list(dict.fromkeys(c.BASE + sel)))
SEL = {'A': ['A_SAG', 'A_su_NDMI', 'A_sg_NDVI'], 'C': ['C_B8_idm_w3'],
       'D': ['D_max_nbr_drop', 'D_cc_n_breaks', 'D_years_continuous_forest', 'D_h_total_loss_px', 'D_ndvi_1985_89', 'D_lt_n_vertices'],
       'B': ['B_SAG_SON', 'B_SAG_phase_sin', 'B_NDRE_phase_sin', 'B_SWIR_phase_sin', 'B_SAG_DJ', 'B_SWIR_phase_cos', 'B_SAG_mean'],
       'F': ['F_VHminusVV_median', 'F_VV_median']}
w12 = list(dict.fromkeys(c.BASE + sum(SEL.values(), [])))
run('Waves 1–2 preview (reference): Baseline + A + B + C + D + F', w12)
for k, v in fam_sel.items(): run(f'Preview: Waves 1–2 preview + Family {k} (main)', list(dict.fromkeys(w12 + v)))
run('Preview only (not Phase 4): all families A–H (main)', list(dict.fromkeys(w12 + sum(fam_sel.values(), []))))
R = pd.DataFrame(out); R.to_csv('w3_models.csv', index=False)
import json; json.dump(dict(selection=sel_log, alert_weights={str(k): v for k, v in info.items()}), open('w3_selection.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 60)
print(json.dumps(sel_log, indent=1, default=str))
print(R[['model', 'n_features', 'MB_AUC', 'MB_AUC_lo', 'MB_rec@1%', 'MB_rec@2%', 'MB_rec@5%', 'MB_weighted_rec@5%', 'MBtop25_FE_at_unweighted_t', 'MBtop25_FE_at_weighted_t',
         'QC_AUC', 'QC_AUC_lo', 'QC_FE_at_MB_t', 'QC_rec_at_MB_t', 'QC_FE_at_weighted_t', 'QC_rec_at_weighted_t']].round(3).to_string())
