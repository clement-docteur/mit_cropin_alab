"""Wave 1 – build Family A (spectral), C (texture) and D (history) features plus context
(for alert-like weights) for MapBiomas dev and QC dev. As-of rule (rule 7):
MapBiomas: history up to and including 2022 (label window end).
QC: history up to the year BEFORE the estimated clearing year (2025 if no clearing detected)."""
import pandas as pd, numpy as np, json, re, os
MB_CUTOFF = int(os.environ.get('MB_CUTOFF', 2022))   # 2012 = sensitivity check (history before the label window)
SUF = '' if MB_CUTOFF == 2022 else f'_mbcut{MB_CUTOFF}'
WIN = 25      # years of Landsat/LandTrendr history used (v2)
WIN_CC = 20   # years of CCDC / Hansen history used (both start ~2000-2001)
mb = pd.read_csv('p0/mb_dev_table.csv'); qc = pd.read_csv('p0/qc_dev_table.csv')
mb['set'] = 'mb'; qc['set'] = 'qc'
mb['cutoff'] = MB_CUTOFF
cy = pd.to_datetime(qc.clear_month, errors='coerce').dt.year
qc['cutoff'] = np.where(cy.notna(), cy - 1, 2025).astype(int)
keys = pd.concat([mb[['id', 'set', 'cutoff']], qc[['id', 'set', 'cutoff']]])

# ---------------- Family A: spectral (from Phase 0 composites) ----------------
def spectral(d):
    o = pd.DataFrame(index=d.index)
    for p in ['su_', 'sg_']:
        b = lambda x: d[p + x] / 1e4
        for x in ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']: o[f'A_{p}{x}'] = b(x)
        o[f'A_{p}NDVI'] = (b('B8') - b('B4')) / (b('B8') + b('B4'))
        o[f'A_{p}EVI'] = 2.5 * (b('B8') - b('B4')) / (b('B8') + 6 * b('B4') - 7.5 * b('B2') + 1)
        o[f'A_{p}NDRE'] = (b('B8A') - b('B5')) / (b('B8A') + b('B5'))
        o[f'A_{p}CIre'] = b('B7') / b('B5') - 1
        o[f'A_{p}NDMI'] = (b('B8') - b('B11')) / (b('B8') + b('B11'))
        o[f'A_{p}NBR'] = (b('B8') - b('B12')) / (b('B8') + b('B12'))
        o[f'A_{p}NDWI'] = (b('B3') - b('B8')) / (b('B3') + b('B8'))
        o[f'A_{p}SWIRratio'] = b('B11') / b('B12')
    o['A_SAG'] = d['SAG']
    return o
# ---------------- Family C: texture (Phase 0 + batch 1b) ----------------
TEXCOLS = [c for c in mb.columns if re.match(r'^(B8|B5|B11|NDVI)_(contrast|ent|idm|asm|corr|var|sd)_w[357]$', c)]
tex = pd.read_csv('gee/w1_tex_glcm_dir.csv')
sn = pd.concat([pd.read_csv(f'gee/{f}') for f in ['w1_snic_mb.csv', 'w1_snic_qc_2025.csv', 'w1_snic_qc_2022_24.csv']])
def texture(d):
    o = d[TEXCOLS].add_prefix('C_').copy()
    t = d[['id', 'set', 'season_start']].merge(tex.drop(columns=['system:index', '.geo', 'label'], errors='ignore'),
                                               on=['id', 'set', 'season_start'], how='left').set_index(d.index)
    for c in t.columns:
        if re.match(r'^(B4|B12)_', c): o['C_' + c] = t[c]
    dd = t[['dirc_B8_d0', 'dirc_B8_d45', 'dirc_B8_d90', 'dirc_B8_d135']]
    o['C_dir_anisotropy'] = dd.max(axis=1) / dd.min(axis=1).replace(0, np.nan)   # H16 rows test
    o['C_dir_contrast_mean'] = dd.mean(axis=1)
    s = d[['id', 'set', 'season_start']].merge(sn.drop(columns=['system:index', '.geo', 'label'], errors='ignore'),
                                               on=['id', 'set', 'season_start'], how='left').set_index(d.index)
    for c in ['seg_sd_B8', 'seg_sd_B11', 'seg_sd_NDVI', 'seg_px']: o['C_' + c] = s[c]
    return o
mb['season_start'] = 2021
# ---------------- Family D: history ----------------
ann = pd.read_csv('gee/w1_landsat_annual.csv')[['id', 'set', 'year', 'NDVI', 'NBR', 'NBR_min']]
def hist_feats(g, cutoff):
    g = g[(g.year <= cutoff)].dropna(subset=['NDVI', 'NBR']).sort_values('year')
    f = {}
    early = g.NDVI[g.year <= 1989]; f['D_ndvi_1985_89'] = early.mean() if len(early) else np.nan   # H7 (fixed calendar years)
    # v2 (reviewer item 5): fixed 25-year window before the cutoff, so MapBiomas and QC use equal spans
    g = g[g.year >= cutoff - (WIN - 1)]
    y, nd, nb, nbmin = g.year.values, g.NDVI.values, g.NBR.values, g.NBR_min.values
    span = WIN
    # disturbance events: NBR falls >= 0.20 below the median of the 2 previous years (H1, H5)
    ev, drops = [], []
    for i in range(2, len(y)):
        ref = np.median(nb[i-2:i]); dr = ref - nb[i]; drops.append(dr)
        if dr >= 0.20 and (not ev or y[i] - ev[-1] > 2): ev.append(y[i])
    f['D_n_events'] = len(ev)
    f['D_years_since_last_event'] = cutoff - ev[-1] if ev else span
    f['D_mean_interval'] = np.mean(np.diff(ev)) if len(ev) >= 2 else span       # H2 rotation length
    f['D_max_nbr_drop'] = max(drops) if drops else np.nan                     # H5 abruptness
    # H3 regrowth: NDVI gain in the 3 years after the last event
    if ev and ev[-1] + 3 <= cutoff:
        e = ev[-1]; f['D_regrowth_3yr'] = nd[y == e + 3].mean() - nd[y == e].mean() if (y == e + 3).any() else np.nan
    else: f['D_regrowth_3yr'] = np.nan
    f['D_post_event_nbr_min'] = nbmin[y == ev[-1]].min() if ev else np.nan     # H8 bareness after clearing
    low = y[nd < 0.6]; f['D_years_continuous_forest'] = cutoff - low.max() if len(low) else span   # H4
    last10 = nd[y > cutoff - 10]
    f['D_ndvi_sd_last10'] = last10.std() if len(last10) > 3 else np.nan                             # H11
    f['D_ndvi_absdiff_last10'] = np.median(np.abs(np.diff(last10))) if len(last10) > 3 else np.nan   # H11
    l15 = (y > cutoff - 15)
    f['D_ndvi_trend_15yr'] = np.polyfit(y[l15], nd[l15], 1)[0] if l15.sum() > 5 else np.nan
    f['D_n_years_obs'] = len(y)
    return f
def landtrendr_feats(arr, cutoff):
    a = np.array(arr); yr, fit, vtx = a[0], a[2], a[3]
    v = yr[(vtx == 1) & (yr <= cutoff) & (yr >= cutoff - (WIN - 1))]; fv = np.array([fit[yr == t][0] for t in v])
    f = {'D_lt_n_vertices': len(v)}
    seg = [(v[i], v[i+1], fv[i+1] - fv[i]) for i in range(len(v) - 1)]   # fitted = -NBR*1000: positive = loss
    dist = [s for s in seg if s[2] >= 100]
    f['D_lt_n_dist'] = len(dist)
    if dist:
        g = max(dist, key=lambda s: s[2]); f['D_lt_greatest_mag'] = g[2]
        f['D_lt_years_since_greatest'] = cutoff - g[1]; f['D_lt_greatest_duration'] = g[1] - g[0]
    else:
        f['D_lt_greatest_mag'] = 0; f['D_lt_years_since_greatest'] = WIN; f['D_lt_greatest_duration'] = 0
    return f
lt = pd.read_csv('gee/w1_landtrendr.csv')
cc = pd.read_csv('gee/w1_ccdc.csv')
# v3 (reviewer item 7): use LandTrendr/CCDC refits whose series end at the as-of cutoff (batch 1c)
# for MapBiomas (2022) and QC points with cutoff < 2025; other QC points keep the batch 1 fit (ends 2025).
if MB_CUTOFF == 2022:
    def swap(orig, cut):
        k = cut.set_index(['id', 'set']).index
        return pd.concat([orig[~orig.set_index(['id', 'set']).index.isin(k)], cut.drop(columns=['cutoff'])], ignore_index=True)
    lt = swap(lt, pd.read_csv('gee/w1c_landtrendr_cut.csv')); cc = swap(cc, pd.read_csv('gee/w1c_ccdc_cut.csv'))
def ccdc_feats(r, cutoff):
    P = lambda k: json.loads(str(r[k]).replace('E', 'e'))
    ts, te, tb, ndc, nbm = P('tStart'), P('tEnd'), P('tBreak'), P('NDVI_coefs'), P('NBR_magnitude')
    br = [(b, m) for b, m in zip(tb, nbm) if b >= cutoff + 1 - WIN_CC and b < cutoff + 1]
    f = {'D_cc_n_breaks': len(br), 'D_cc_years_since_break': (cutoff + 1 - br[-1][0]) if br else WIN_CC,
         'D_cc_last_break_nbr_mag': br[-1][1] if br else 0.0}
    t = cutoff + 0.5; k = None
    for i in range(len(ts)):
        if ts[i] <= t: k = i
    if k is None: f.update(D_cc_ndvi_amp=np.nan, D_cc_ndvi_level=np.nan)
    else:
        c = ndc[k]; f['D_cc_ndvi_amp'] = float(np.hypot(c[2], c[3])); f['D_cc_ndvi_level'] = c[0] + c[1] * t
    return f
han = pd.read_csv('gee/w1_hansen.csv'); han = han[han.histogram.notna()]
def hansen_feats(hs, cutoff):
    d = {int(k): float(v) for k, v in re.findall(r'(\d+)=([\d.Ee-]+)', hs)}
    d = {2000 + k: v for k, v in d.items() if 0 < k and cutoff - (WIN_CC - 1) <= 2000 + k <= cutoff}
    tot = 78.5   # pixels in a 150 m radius at 30 m
    return {'D_h_max_year_px': max(d.values()) if d else 0.0, 'D_h_n_loss_years': len(d),
            'D_h_total_loss_px': sum(d.values())}   # weighted 30 m pixel counts within ~150 m (not a true share)
annG = {k: g for k, g in ann.groupby(['id', 'set'])}
ltI = lt.set_index(['id', 'set']); ccI = cc.set_index(['id', 'set']); hI = han.set_index(['id', 'set'])
rows = []
for _, k in keys.iterrows():
    key = (k.id, k.set); f = {'id': k.id, 'set': k.set}
    f.update(hist_feats(annG[key], k.cutoff))
    f.update(landtrendr_feats(json.loads(ltI.loc[key, 'first']), k.cutoff))
    f.update(ccdc_feats(ccI.loc[key], k.cutoff))
    f.update(hansen_feats(hI.loc[key, 'histogram'], k.cutoff))
    rows.append(f)
D = pd.DataFrame(rows)
# ---------------- context (alert-like weights only) ----------------
ctx = pd.read_csv('gee/w1_context.csv').drop(columns=['system:index', '.geo', 'label'])
def build(d):
    base = d[['id', 'set', 'label', 'block', 'cutoff', 'su_n_clear', 'chm_mean_r15', 'chm_stdDev_r15', 'nfw', 'SAG']
             + [c for c in ['agree', 'border', 'state', 'biome', 'weight', 'season_start', 'clear_month'] if c in d.columns]]
    out = pd.concat([base, spectral(d), texture(d)], axis=1)
    return out.merge(D, on=['id', 'set'], how='left').merge(ctx, on=['id', 'set'], how='left')
qc['block'] = np.floor(qc.lon/0.25).astype(int).astype(str)+'_'+np.floor(qc.lat/0.25).astype(int).astype(str)
MB = build(mb); QC = build(qc)
MB.to_csv(f'w1_mb_dev_features{SUF}.csv', index=False); QC.to_csv(f'w1_qc_dev_features{SUF}.csv', index=False)
fam = {p: [c for c in MB.columns if c.startswith(p + '_')] for p in 'ACD'}
print({p: len(v) for p, v in fam.items()}, MB.shape, QC.shape)
na = MB[sum(fam.values(), [])].isna().mean(); print('MB missing share >0:', na[na > 0].round(3).to_dict())
na = QC[sum(fam.values(), [])].isna().mean(); print('QC missing share >0:', na[na > 0].round(3).to_dict())
print(QC.cutoff.value_counts().to_dict())
