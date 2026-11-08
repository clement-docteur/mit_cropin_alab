"""Phase 0 – prepare labelled point sets (MapBiomas + Cropin QC).
Rules: Workflow Plan v4.1 §5–6. Seed fixed for reproducibility."""
import json, numpy as np, pandas as pd
from p0_dbf import read_dbf
SEED = 2026
ROOT = '../..'
OUT = '../results/phase0'
import os; os.makedirs(OUT, exist_ok=True)
log = []
def note(s): print(s); log.append(s)

# ---------------- MapBiomas ----------------
df = read_dbf(f'{ROOT}/mapbiomas_85k_points_validation/mapbiomas_85k_points_validation.dbf')
note(f'MapBiomas file rows: {len(df)}')
df = df[df.SIGLA.isin(['PR', 'SC', 'RS'])].copy()
note(f'Rows in PR/SC/RS: {len(df)}')
# Data check: fields are swapped (LON holds latitude)
df['lat'], df['lon'] = df['LON'], df['LAT']
assert df.lat.between(-34, -22).all() and df.lon.between(-58, -47).all(), 'coordinates outside southern Brazil'
note('Coordinate check passed after swapping LON/LAT')
yrs = list(range(2013, 2023))
C = df[[f'CLASS_{y}' for y in yrs]]
note(f'Null class values 2013–2022: {int(C.isna().sum().sum())}')
df['label'] = np.where((C == 3).all(axis=1), 0, np.where((C == 9).all(axis=1), 1, -1))
note(f'Stable forest (class 3, 2013–2022): {(df.label==0).sum()}; stable plantation (class 9, 2013–2022): {(df.label==1).sum()}')
mb = df[df.label >= 0].copy()
# class history before 2013 (kept for description only — NEVER a feature, rule 1)
pre = mb[[f'CLASS_{y}' for y in range(1985, 2013)]]
mb['forest_all_1985_2022'] = ((pre == 3).all(axis=1) & (mb.label == 0)).astype(int)
mb['agree'] = mb['COUNT_2022']; mb['border'] = mb['BORDA_2022']
mb['state'] = mb['SIGLA']; mb['biome'] = mb['BIOMA_250K']; mb['weight'] = mb['PESO_AMOS']
mb['id'] = mb['TARGETID'].astype(int)
note(f'Border values: {mb.border.value_counts().to_dict()}; agree values: {mb.agree.value_counts().to_dict()}')

# Dedup ~100 m (equirectangular distance is fine at this scale)
def dedup(d, radius_m=100):
    lat0 = np.deg2rad(d.lat.mean())
    x = d.lon.values * 111320 * np.cos(lat0); y = d.lat.values * 110540
    keep = np.ones(len(d), bool); order = np.argsort(d.id.values)
    pairs = 0
    for a_i, i in enumerate(order):
        if not keep[i]: continue
        dist = np.hypot(x - x[i], y - y[i])
        close = np.where((dist < radius_m) & (np.arange(len(d)) != i) & keep)[0]
        for j in close:
            pairs += 1; keep[j] = False
            if d.label.values[j] != d.label.values[i]: keep[i] = False  # conflicting labels: drop both
    return d[keep], pairs
mb, npairs = dedup(mb.reset_index(drop=True))
note(f'Pairs within 100 m removed: {npairs}; MapBiomas points after dedup: {len(mb)} (forest {(mb.label==0).sum()}, plantation {(mb.label==1).sum()})')

# 0.5° blocks + lockbox (~20% of blocks, stratified by state of block majority)
mb['block'] = (np.floor(mb.lon / 0.5).astype(int).astype(str) + '_' + np.floor(mb.lat / 0.5).astype(int).astype(str))
blk = mb.groupby('block').agg(state=('state', lambda s: s.mode()[0]), n=('id', 'size')).reset_index()
rng = np.random.default_rng(SEED)
lock = []
for st, g in blk.groupby('state'):
    k = int(round(0.2 * len(g)))
    lock += list(rng.choice(g.block.values, size=k, replace=False))
mb['split'] = np.where(mb.block.isin(lock), 'lockbox', 'dev')
note(f'Blocks: {len(blk)}; lockbox blocks: {len(lock)}')
t = mb.groupby(['split', 'label']).size().unstack(); t.columns = ['forest', 'plantation']
note('Split counts:\n' + t.to_string())
note('Dev points by state:\n' + mb[mb.split=='dev'].groupby(['state','label']).size().unstack().to_string())
cols = ['id','lon','lat','label','agree','border','state','biome','weight','block','split','forest_all_1985_2022']
mb[cols].to_csv(f'{OUT}/mb_points_all.csv', index=False)
mb[mb.split=='lockbox'][cols].to_csv(f'{OUT}/LOCKBOX_mb_points_DO_NOT_OPEN.csv', index=False)
mb[mb.split=='dev'][cols].to_csv(f'{OUT}/mb_points_dev.csv', index=False)

# ---------------- Cropin QC ----------------
g = json.load(open(f'{ROOT}/cropin_qc_data/cropin_brazil_qc_forest_plantation_1.geojson'))
rows = []
for i, f in enumerate(g['features']):
    geom = f['geometry']; ring = geom['coordinates'][0] if geom['type']=='Polygon' else geom['coordinates'][0][0]
    xs = np.array([p[0] for p in ring]); ys = np.array([p[1] for p in ring])
    lat0 = np.deg2rad(ys.mean())
    X = xs*111320*np.cos(lat0); Y = ys*110540
    area = 0.5*abs(np.dot(X, np.roll(Y, 1)) - np.dot(Y, np.roll(X, 1)))
    rows.append(dict(id=i, lon=xs[:-1].mean(), lat=ys[:-1].mean(), qc_label=f['properties'].get('qc_label'), area_ha=area/1e4, geomtype=geom['type']))
qc = pd.DataFrame(rows)
note(f'QC polygons: {len(qc)}; labels: {qc.qc_label.value_counts().to_dict()}; geometry types: {qc.geomtype.value_counts().to_dict()}')
note(f'QC area ha: min {qc.area_ha.min():.3f}, median {qc.area_ha.median():.3f}, max {qc.area_ha.max():.3f}')
qc['label'] = (qc.qc_label.str.lower().str.startswith('plant')).astype(int)
qc['state'] = None
qc, qpairs = dedup(qc)
note(f'QC pairs within 100 m removed: {qpairs}; QC after dedup: {len(qc)} (forest {(qc.label==0).sum()}, plantation {(qc.label==1).sum()})')
for size in [0.5, 0.25, 0.1]:
    b = (np.floor(qc.lon/size).astype(int).astype(str)+'_'+np.floor(qc.lat/size).astype(int).astype(str))
    note(f'QC blocks at {size}°: {b.nunique()} (largest block holds {b.value_counts().iloc[0]} points)')
qc.to_csv(f'{OUT}/qc_points_prepped.csv', index=False)
open(f'{OUT}/p0_prepare_log.txt','w').write('\n'.join(log))
