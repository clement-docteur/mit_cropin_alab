#!/usr/bin/env python3
"""Step 2: build the point table: MapBiomas validation points (PR/SC/RS, non-edge, forest/plantation in 2022;
the same 1,831 rows as data_exploration/cropin_model_features_2022.csv) + Cropin QC polygon representative points,
with the nearest Landsat WRS-2 path/rows.

Usage: python build_points.py --mb-shp <mapbiomas_85k_points_validation.shp> --qc <cropin_qc.geojson>
                              --index <landsat_index_south.csv from filter_index.py> --out <data_dir>
"""
import argparse
import os
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--mb-shp", required=True)
ap.add_argument("--qc", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--team-features", default=os.path.join(REPO, "data_exploration", "cropin_model_features_2022.csv"))
ap.add_argument("--out", required=True)
args = ap.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)
yrs = list(range(1985, 2023))
g = gpd.read_file(args.mb_shp)
for y in yrs:
    g[f'CLASS_{y}'] = pd.to_numeric(g[f'CLASS_{y}'], errors='coerce')
s = g[g.SIGLA.isin(['PR', 'SC', 'RS']) & g.CLASS_2022.isin([3, 9]) & (g.BORDA_2022 == 0)].copy()
s['lon'] = s.geometry.x; s['lat'] = s.geometry.y
f = pd.read_csv(args.team_features)
d, i = cKDTree(np.c_[s.lon, s.lat]).query(np.c_[f.lon, f.lat])
assert d.max() < 1e-4 and len(set(i)) == len(f)
s = s.iloc[i].copy()
s['pt_id'] = f['pt_id'].to_numpy()
val = pd.DataFrame({'pid': 'V' + s.TARGETID.astype(int).astype(str), 'dataset': 'val',
                    'pt_id': s.pt_id.to_numpy(), 'TARGETID': s.TARGETID.astype(int).to_numpy(),
                    'lon': s.lon.to_numpy(), 'lat': s.lat.to_numpy(),
                    'label': np.where(s.CLASS_2022.to_numpy() == 9, 1, 0),
                    'CD_MUN': s.CD_MUN.astype(int).to_numpy(), 'state': s.SIGLA.to_numpy()})
hist = s[['TARGETID'] + [f'CLASS_{y}' for y in yrs] + [f'COUNT_{y}' for y in [2020, 2022]] +
         ['BIOMA_250K', 'DECLIVIDAD', 'NM_MUN']].copy()
hist['pid'] = 'V' + hist.TARGETID.astype(int).astype(str)
q = gpd.read_file(args.qc)
rp = q.geometry.representative_point()
area = q.to_crs(32722).area
qc = pd.DataFrame({'pid': [f'Q{k:03d}' for k in range(len(q))], 'dataset': 'qc', 'pt_id': '',
                   'TARGETID': -1, 'lon': rp.x.to_numpy(), 'lat': rp.y.to_numpy(),
                   'label': (q.qc_label == 'plantation').astype(int).to_numpy(),
                   'CD_MUN': -1, 'state': '', 'qc_area_m2': area.to_numpy()})
pts = pd.concat([val, qc], ignore_index=True)
# path/row centres (daytime rows) from the index
ix = pd.read_csv(args.index)
ix = ix[ix.WRS_ROW <= 122]
ix['clat'] = (ix.NORTH_LAT + ix.SOUTH_LAT) / 2; ix['clon'] = (ix.WEST_LON + ix.EAST_LON) / 2
pr = ix[ix.DATA_TYPE == 'L1TP'].groupby(['WRS_PATH', 'WRS_ROW']).agg(clat=('clat', 'median'), clon=('clon', 'median'), n=('SCENE_ID', 'size')).reset_index()
pr = pr[pr.n >= 50]
# distance (km) from each point to each path/row centre (equirectangular)
R = 6371.0
lat0 = np.deg2rad(pts.lat.to_numpy())[:, None]
dx = np.deg2rad(pts.lon.to_numpy()[:, None] - pr.clon.to_numpy()[None, :]) * np.cos(lat0) * R
dy = np.deg2rad(pts.lat.to_numpy()[:, None] - pr.clat.to_numpy()[None, :]) * R
dist = np.hypot(dx, dy)
order = np.argsort(dist, axis=1)
pts['pr1'] = [f"{pr.WRS_PATH.iloc[o[0]]:03d}{pr.WRS_ROW.iloc[o[0]]:03d}" for o in order]
pts['d1_km'] = dist[np.arange(len(pts)), order[:, 0]]
pts['pr2'] = [f"{pr.WRS_PATH.iloc[o[1]]:03d}{pr.WRS_ROW.iloc[o[1]]:03d}" for o in order]
pts['d2_km'] = dist[np.arange(len(pts)), order[:, 1]]
print(pts.dataset.value_counts().to_dict(), pts.groupby('dataset').label.mean().round(3).to_dict())
print('d1 km quantiles', np.percentile(pts.d1_km, [50, 90, 99, 100]).round(1))
print('d2 km quantiles', np.percentile(pts.d2_km, [10, 50, 90]).round(1), ' share d2<90km:', (pts.d2_km < 90).mean().round(3))
print('primary PRs:', pts.pr1.nunique(), ' any PR (d<90):', len(set(pts.pr1) | set(pts.loc[pts.d2_km < 90, 'pr2'])))
pts.to_csv(f'{OUT}/points.csv', index=False)
hist.to_csv(f'{OUT}/val_label_history.csv', index=False)
pr.to_csv(f'{OUT}/pathrow_centres.csv', index=False)
