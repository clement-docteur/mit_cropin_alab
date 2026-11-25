"""Master v34, step 5: compute the v3 features for the November QC points from the v34 GEE exports.
Every formula is copied from the script that made the v3 column (named in each section). Nothing is refit
on labels. Two inputs are optional and their columns stay empty until the files exist:
  v34_hansen.csv        -> nc_D_h_total_loss_px
  v34_b6_monthly_qc.csv -> mm_B6_win   (re-export; the first B6 export came back empty)
Harvest-cycle levels (closed-canopy and open-land NBR) were set on all v3 points and not saved; they are
passed in (see 06_check_repeats.py, which recovers them from the repeats of v3 polygons).
Output: 'Master Data/v34_work/qcnov_features.csv' (git-ignored)."""
import json, os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = f'{HERE}/..'
W = f'{ROOT}/Master Data/v34_work'; G = f'{W}/gee'
DROPC = ['system:index', '.geo', 'label', 'cutoff', 'season_start', 'set']


def rd(name):
    return pd.read_csv(f'{G}/{name}.csv').drop(columns=DROPC, errors='ignore')


def sag(b2, b3, b4):   # Cropin's SAG (p0_build_tables.sag / w2_features), reflectance 0-1
    xb, xg, xr = 0.4966, 0.560, 0.6645
    v1 = np.stack([np.full(len(b2), xb - xg), b2 - b3]); v2 = np.stack([np.full(len(b2), xr - xg), b4 - b3])
    return np.degrees(np.arccos(np.clip((v1 * v2).sum(0) / (np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0)), -1, 1)))


# ======================= w1_features.py (Family D), copied =======================
WIN, WIN_CC = 25, 20


def hist_feats(g, cutoff):
    g = g[(g.year <= cutoff)].dropna(subset=['NDVI', 'NBR']).sort_values('year')
    f = {}
    early = g.NDVI[g.year <= 1989]; f['D_ndvi_1985_89'] = early.mean() if len(early) else np.nan
    g = g[g.year >= cutoff - (WIN - 1)]
    y, nd, nb = g.year.values, g.NDVI.values, g.NBR.values
    span = WIN
    ev, drops = [], []
    for i in range(2, len(y)):
        ref = np.median(nb[i - 2:i]); dr = ref - nb[i]; drops.append(dr)
        if dr >= 0.20 and (not ev or y[i] - ev[-1] > 2): ev.append(y[i])
    f['D_max_nbr_drop'] = max(drops) if drops else np.nan
    f['D_years_since_last_event'] = cutoff - ev[-1] if ev else span
    low = y[nd < 0.6]; f['D_years_continuous_forest'] = cutoff - low.max() if len(low) else span
    return f


def landtrendr_feats(arr, cutoff):
    a = np.array(arr); yr, vtx = a[0], a[3]
    v = yr[(vtx == 1) & (yr <= cutoff) & (yr >= cutoff - (WIN - 1))]
    return {'D_lt_n_vertices': len(v)}


def ccdc_feats(r, cutoff):
    P = lambda k: json.loads(str(r[k]).replace('E', 'e'))
    tb, nbm = P('tBreak'), P('NBR_magnitude')
    br = [(b, m) for b, m in zip(tb, nbm) if b >= cutoff + 1 - WIN_CC and b < cutoff + 1]
    return {'D_cc_n_breaks': len(br)}


def hansen_feats(hs, cutoff):
    d = {int(k): float(v) for k, v in re.findall(r'(\d+)=([\d.Ee-]+)', hs)}
    d = {2000 + k: v for k, v in d.items() if 0 < k and cutoff - (WIN_CC - 1) <= 2000 + k <= cutoff}
    return {'D_h_total_loss_px': sum(d.values())}


def family_d(pts):
    ann = rd('v34_landsat_annual').assign(set='qc')
    annG = {i: g for i, g in ann.groupby('id')}
    lt = rd('v34_landtrendr_cut').set_index('id'); cc = rd('v34_ccdc_cut').set_index('id')
    han = None
    if os.path.exists(f'{G}/v34_hansen.csv'):
        h = rd('v34_hansen'); han = h[h.histogram.notna()].set_index('id')
    rows = []
    for k in pts.itertuples():
        f = {'id': k.id}
        f.update(hist_feats(annG[k.id], k.asof_year))
        f.update(landtrendr_feats(json.loads(lt.loc[k.id, 'first']), k.asof_year))
        f.update(ccdc_feats(cc.loc[k.id], k.asof_year))
        f.update(hansen_feats(han.loc[k.id, 'histogram'], k.asof_year) if han is not None else {'D_h_total_loss_px': np.nan})
        rows.append(f)
    return pd.DataFrame(rows).rename(columns=lambda c: c if c == 'id' else 'nc_' + c)


# ======================= w2_features.py (Family B harmonics, Family F radar), copied =======================
def s2_monthly():
    S = rd('v34_s2_monthly_qc').dropna(subset=['B8'])
    b = lambda x: S[x] / 1e4
    S['NDVI'] = (b('B8') - b('B4')) / (b('B8') + b('B4')); S['NDMI'] = (b('B8') - b('B11')) / (b('B8') + b('B11'))
    S['NDRE'] = (b('B8A') - b('B5')) / (b('B8A') + b('B5')); S['SWIR'] = b('B11'); S['B8A_r'] = b('B8A')
    S['SAG'] = sag(b('B2'), b('B3'), b('B4'))
    S['t'] = pd.to_datetime(S.month); S['m'] = S.t.dt.month; S['mo'] = S.m; S['yr'] = S.t.dt.year
    S['tyr'] = S.t.dt.year + (S.m - 0.5) / 12
    return S


def family_b(S):
    rows = []
    for i, g in S.groupby('id'):
        f = {'id': i, 'B_n_months': len(g)}
        X = np.c_[np.ones(len(g)), np.cos(2 * np.pi * g.tyr), np.sin(2 * np.pi * g.tyr)]
        for k in ['NDRE', 'SWIR', 'SAG']:
            if len(g) >= 12:
                c, *_ = np.linalg.lstsq(X, g[k].values, rcond=None)
                ph = np.arctan2(c[2], c[1]); f[f'B_{k}_phase_cos'] = np.cos(ph); f[f'B_{k}_phase_sin'] = np.sin(ph)
        f['B_SAG_SON'] = g[g.m.isin([9, 10, 11])].SAG.median()
        rows.append(f)
    B = pd.DataFrame(rows)[['id', 'B_n_months', 'B_SAG_SON', 'B_SAG_phase_sin', 'B_NDRE_phase_sin', 'B_SWIR_phase_sin', 'B_SWIR_phase_cos']]
    return B.rename(columns=lambda c: c if c in ('id', 'B_n_months') else 'nc_' + c)


def three_seasons(S):
    """build_master.py (3-yr NDMI), build_master_v2.py (3-yr Oct-Jan NDVI), meera_features.py (mm_), copied."""
    def med(sub, key, var):
        y = sub.groupby(['id', key])[var].agg(['median', 'size']).reset_index()
        return y[y['size'] >= 2].groupby('id')['median'].median()
    S = S.copy(); S['ss'] = np.where(S.mo == 12, S.yr, S.yr - 1)               # summer Dec-Mar season key
    S['sg'] = np.where(S.mo >= 10, S.yr, S.yr - 1)                             # Oct-Jan season key
    su, wi, sg = S[S.mo.isin([12, 1, 2, 3])], S[S.mo.between(4, 9)], S[S.mo.isin([10, 11, 12, 1])]
    return pd.DataFrame({'nc_su3_NDMI': med(su, 'ss', 'NDMI'), 'nc_wi3_NDMI': med(wi, 'yr', 'NDMI'),
                         'nc_sg3_NDVI': med(sg, 'sg', 'NDVI'), 'mm_SAG_win': med(wi, 'yr', 'SAG'),
                         'mm_NDVI_win': med(wi, 'yr', 'NDVI'), 'mm_B8A_win': med(wi, 'yr', 'B8A_r')}).reset_index()


def b6_winter():
    """build_master_v2.py / search_v2.py: monthly B6 (Apr-Sep), median of yearly medians with >= 2 months."""
    f = f'{G}/v34_b6_monthly_qc.csv'
    if not os.path.exists(f): return None
    B = pd.read_csv(f).dropna(subset=['B6'])
    d = pd.to_datetime(B.month); B = B[d.dt.month.between(4, 9)].copy(); B['yr'] = d.dt.year; B['B6'] = B.B6 / 1e4
    y = B.groupby(['id', 'yr']).B6.agg(['median', 'size']).reset_index()
    return y[y['size'] >= 2].groupby('id')['median'].median().rename('mm_B6_win').reset_index()


def family_f():
    s1 = rd('v34_s1')
    return pd.DataFrame({'id': s1.id, 'nc_F_VH_median': s1.VH_median, 'nc_F_VHminusVV_median': s1.VHminusVV_median})


# ======================= w3_features.py (E), w3 Hansen tc, Phase 0 (A_SAG, nfw, Meta CHM), copied =======================
def family_e_h_p0(pts):
    eth = rd('v34_eth'); htc = rd('v34_hansen_tc').rename(columns={'first': 'nc_hansen_tc2000'})
    E = pd.DataFrame({'id': eth.id, 'nc_E_eth_h': eth.eth_h, 'nc_E_eth_sd50': eth.eth_h_sd50})
    nfw = rd('v34_nfw').rename(columns={'nfw': 'nc_nfw'})
    chm = pd.read_csv(f'{G}/v34_chm.csv'); chm = chm[chm.radius_m == 15]
    C = pd.DataFrame({'id': chm.id, 'nc_chm_mean_r15': chm['mean'], 'nc_chm_stdDev_r15': chm['stdDev']})
    qs = pd.read_csv(f'{G}/v34_qc_s2_seasons.csv')
    qs = pts[['id', 'season_start']].merge(qs, on=['id', 'season_start'], how='left')
    A = pd.DataFrame({'id': qs.id, 'nc_A_SAG': sag(qs.sg_B2 / 1e4, qs.sg_B3 / 1e4, qs.sg_B4 / 1e4),
                      'tree_ndvi_summer': qs.su_NDVI})
    return E.merge(htc[['id', 'nc_hansen_tc2000']], on='id').merge(nfw[['id', 'nc_nfw']], on='id') \
            .merge(C, on='id').merge(A, on='id')


# ======================= rebuild_harvest.py (rh_), notebook functions =======================
YEARS = np.arange(1985, 2026)
_g = dict(np=np, pd=pd, YEARS=YEARS)
_cells = json.load(open(f'{ROOT}/harvest_cycles/harvest_cycles_findings.ipynb', encoding='utf-8'))['cells']
_src = lambda i: ''.join(_cells[i]['source'])
exec(compile(_src(8), 'nb_cell_8', 'exec'), _g)
_c10 = _src(10); exec(compile(_c10[:_c10.index('def all_features')], 'nb_cell_10', 'exec'), _g)


def nbr_series(pts):
    L = rd('v34_landsat_annual'); L['year'] = L.year.astype(int)
    S = L.pivot_table(index='id', columns='year', values='NBR').reindex(columns=YEARS)
    for i, y in pts.set_index('id').asof_year.items():
        S.loc[i, [c for c in YEARS if c > y]] = np.nan
    return S


def family_rh(S, canopy, open_level):
    rows = []
    for i, row in S.iterrows():
        f = _g['pixel_features'](row, canopy, open_level)
        rows.append({'id': i, 'rh_n_cycles': f.get('n_cycles'), 'rh_s_std': f.get('s_std'),
                     'rh_s_frac_below_forest': f.get('s_frac_below_forest'), 'rh_abrupt_start': f.get('abrupt_start'),
                     'rh_forest_since_1985': f.get('forest_since_1985'), 'rh_max_drop': f.get('max_drop')})
    return pd.DataFrame(rows)


# ======================= land_history.py (lh_), copied =======================
NAT = {3, 4, 5, 6, 49}
FARM = {15, 18, 19, 20, 21, 35, 36, 39, 40, 41, 46, 47, 48, 62}
OPEN = FARM | {12, 11, 23, 24, 25, 29, 30, 50}


def family_lh(pts):
    H = rd('v34_mapbiomas_history'); asof = pts.set_index('id').asof_year
    rows = []
    for r in H.itertuples():
        last = int(min(asof[r.id], 2023))
        seq = [getattr(r, f'classification_{y}') for y in range(1985, last + 1)]
        seq = [int(v) if pd.notna(v) else -1 for v in seq]
        run = 0
        for v in reversed(seq):
            if v in NAT: run += 1
            else: break
        rows.append({'id': r.id, 'lh_natural_run_years': run, 'lh_ever_open': int(any(v in OPEN for v in seq)),
                     'lh_n_changes': sum(a != b for a, b in zip(seq[:-1], seq[1:]))})
    return pd.DataFrame(rows)


def alphaearth():
    a = rd('v34_alphaearth')
    return a[['id'] + [f'A{i:02d}' for i in range(64)]].rename(columns={f'A{i:02d}': f'ae_A{i:02d}' for i in range(64)}), \
        a.emb_year.value_counts().to_dict()


def build(canopy, open_level):
    pts = pd.read_csv(f'{W}/qcnov_points_asof.csv')
    S = s2_monthly()
    F = pts.merge(family_d(pts), on='id', how='left').merge(family_b(S), on='id', how='left') \
           .merge(three_seasons(S), on='id', how='left').merge(family_f(), on='id', how='left') \
           .merge(family_e_h_p0(pts), on='id', how='left') \
           .merge(family_rh(nbr_series(pts), canopy, open_level), on='id', how='left') \
           .merge(family_lh(pts), on='id', how='left')
    b6 = b6_winter()
    F = F.merge(b6, on='id', how='left') if b6 is not None else F.assign(mm_B6_win=np.nan)
    assert len(F) == len(pts) == F.id.nunique()
    return F


if __name__ == '__main__':
    canopy, open_level = (float(a) for a in sys.argv[1:3])
    F = build(canopy, open_level)
    F.to_csv(f'{W}/qcnov_features.csv', index=False)
    AE, years = alphaearth(); AE.to_csv(f'{W}/qcnov_alphaearth.csv', index=False)
    print(F.shape, '| AlphaEarth year used:', years)
    na = F.isna().sum(); print('columns with missing values:', na[na > 0].to_dict())
