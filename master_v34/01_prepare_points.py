"""Master v34, step 1: points for Cropin's November QC polygons (sealed blind test).

Same rules as v3 (feature_exploration/scripts/p0_prepare.py, p0_qc_split.py):
  - point = mean of the vertices of the first ring (first part for MultiPolygons)
  - area from that ring, equirectangular projection
  - pairs within 100 m are deduplicated (conflicting labels: both dropped)
  - block = 0.25 degree cell
New for v34:
  - id = 1000 + GeoJSON index (v3 QC ids are 0-432), uid = qcnov_<GeoJSON index>
  - split = 'nov' for all rows: a sealed test, never used for tuning
  - repeats of v3 QC polygons (point within 100 m of a v3 QC point, or inside a v3 QC polygon)
    are flagged and excluded from the sealed test, because models have seen them.
Outputs go to the git-ignored 'Master Data/v34_work/'. Prints aggregates only.
"""
import json, os
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
NEW = f'{ROOT}/CROPIN-MIT-DATA/cropin_qc_data/cropin_qc_data_forest_plantation_nov.geojson'
OLD = f'{ROOT}/CROPIN-MIT-DATA/cropin_qc_data/cropin_brazil_qc_forest_plantation_1.geojson'
V3 = f'{ROOT}/Master Data/master_features_v3.csv'
OUT = f'{ROOT}/Master Data/v34_work'
os.makedirs(OUT, exist_ok=True)


def first_ring(geom):
    return geom['coordinates'][0] if geom['type'] == 'Polygon' else geom['coordinates'][0][0]


def rings(geom):   # outer rings of every part (for the inside-old-polygon check)
    return [geom['coordinates'][0]] if geom['type'] == 'Polygon' else [p[0] for p in geom['coordinates']]


def inside(x, y, ring):   # ray casting
    n, ins = len(ring), False
    for i in range(n - 1):
        (x1, y1), (x2, y2) = ring[i][:2], ring[i + 1][:2]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            ins = not ins
    return ins


def points(path, id_offset):
    rows = []
    for i, f in enumerate(json.load(open(path))['features']):
        ring = first_ring(f['geometry'])
        xs = np.array([p[0] for p in ring]); ys = np.array([p[1] for p in ring])
        lat0 = np.deg2rad(ys.mean()); X = xs * 111320 * np.cos(lat0); Y = ys * 110540
        area = 0.5 * abs(np.dot(X, np.roll(Y, 1)) - np.dot(Y, np.roll(X, 1)))
        rows.append(dict(id=id_offset + i, gj_index=i, lon=xs[:-1].mean(), lat=ys[:-1].mean(),
                         qc_label=f['properties'].get('qc_label'), area_ha=area / 1e4,
                         geomtype=f['geometry']['type'], n_parts=len(rings(f['geometry']))))
    return pd.DataFrame(rows)


def dist_m(lon, lat, lon2, lat2):
    lat0 = np.deg2rad(lat)
    return np.hypot((lon2 - lon) * 111320 * np.cos(lat0), (lat2 - lat) * 110540)


def dedup(d, radius_m=100):   # identical to p0_prepare.dedup
    lat0 = np.deg2rad(d.lat.mean())
    x = d.lon.values * 111320 * np.cos(lat0); y = d.lat.values * 110540
    keep = np.ones(len(d), bool); order = np.argsort(d.id.values); pairs = 0
    for i in order:
        if not keep[i]: continue
        dist = np.hypot(x - x[i], y - y[i])
        for j in np.where((dist < radius_m) & (np.arange(len(d)) != i) & keep)[0]:
            pairs += 1; keep[j] = False
            if d.label.values[j] != d.label.values[i]: keep[i] = False
    return d[keep], pairs


q = points(NEW, 1000)
q['label'] = q.qc_label.str.lower().str.startswith('plant').astype(int)
print(f'New polygons: {len(q)}; labels {q.qc_label.value_counts().to_dict()}; '
      f'types {q.geomtype.value_counts().to_dict()}; multipart {int((q.n_parts > 1).sum())}')
assert q.lat.between(-34, -22).all() and q.lon.between(-58, -47).all(), 'outside southern Brazil'

# ---- repeats of v3 QC polygons ----
old_feats = json.load(open(OLD))['features']
v3 = pd.read_csv(V3)
v3q = v3[v3.set == 'qc'][['id', 'lon', 'lat', 'label', 'split']]
near, inpoly, same_label = [], [], []
for r in q.itertuples():
    d = dist_m(r.lon, r.lat, v3q.lon.values, v3q.lat.values)
    k = int(np.argmin(d)); near.append(d[k] < 100)
    hits = [j for j, f in enumerate(old_feats) if any(inside(r.lon, r.lat, rg) for rg in rings(f['geometry']))]
    inpoly.append(len(hits) > 0)
    lab_old = [int(old_feats[j]['properties']['qc_label'].lower().startswith('plant')) for j in hits]
    same_label.append(None if not hits else all(l == r.label for l in lab_old))
q['near_v3_100m'] = near; q['inside_v3_polygon'] = inpoly; q['v3_same_label'] = same_label
q['repeat_of_v3'] = q.near_v3_100m | q.inside_v3_polygon
print(f'Within 100 m of a v3 QC point: {int(q.near_v3_100m.sum())}; inside a v3 QC polygon: {int(q.inside_v3_polygon.sum())}; '
      f'either: {int(q.repeat_of_v3.sum())}')
rep = q[q.inside_v3_polygon]
if len(rep): print(f'  of those inside a v3 polygon, same label as v3: {int((rep.v3_same_label == True).sum())}, different: {int((rep.v3_same_label == False).sum())}')

# ---- dedup within the new set, then split ----
q, npairs = dedup(q.reset_index(drop=True))
print(f'Pairs within 100 m inside the new set removed: {npairs}; left: {len(q)}')
q['block'] = np.floor(q.lon / 0.25).astype(int).astype(str) + '_' + np.floor(q.lat / 0.25).astype(int).astype(str)
q['split'] = np.where(q.repeat_of_v3, 'nov_repeat', 'nov')
q['uid'] = 'qcnov_' + q.gj_index.astype(str)
q['set'] = 'qc'
q['state'] = None
print(q.groupby(['split', 'label']).size().unstack(fill_value=0).rename(columns={0: 'forest', 1: 'plantation'}).to_string())
print(f'Blocks: {q.block.nunique()} (largest holds {q.block.value_counts().iloc[0]}); '
      f'blocks shared with v3 QC: {len(set(q.block) & set(v3[v3.set == "qc"].block))}')
print(f'Area ha: median {q.area_ha.median():.2f}, min {q.area_ha.min():.3f}, max {q.area_ha.max():.1f}')
q.to_csv(f'{OUT}/qcnov_points_all.csv', index=False)
print('written', f'{OUT}/qcnov_points_all.csv')
