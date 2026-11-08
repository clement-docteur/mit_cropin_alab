#!/usr/bin/env python3
"""Step 5: sample Hansen GFC v1.12 (treecover2000, gain 2000-2012, lossyear 2001-2024) at the points (centre pixel + 3x3).
Usage: python sample_hansen.py <data_dir with points.csv>"""
import os, sys, math
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                  GDAL_HTTP_MAX_RETRY='8', GDAL_HTTP_RETRY_DELAY='2')
import numpy as np, pandas as pd, rasterio
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor
D = sys.argv[1]
pts = pd.read_csv(f'{D}/points.csv')
URL = 'https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/Hansen_GFC-2024-v1.12_{l}_{t}.tif'
def tile(lon, lat):
    top = math.ceil(lat / 10) * 10; left = math.floor(lon / 10) * 10
    return f"{abs(top):02d}{'N' if top >= 0 else 'S'}_{abs(left):03d}{'E' if left >= 0 else 'W'}"
pts['tile'] = [tile(a, b) for a, b in zip(pts.lon, pts.lat)]
print(pts.tile.value_counts().to_dict())
LAYERS = ['treecover2000', 'gain', 'lossyear']
def work(args):
    t, lyr = args
    sub = pts[pts.tile == t]
    out = {}
    with rasterio.open('/vsicurl/' + URL.format(l=lyr, t=t)) as s:
        for r in sub.itertuples():
            row, col = s.index(r.lon, r.lat)
            w = s.read(1, window=Window(col - 1, row - 1, 3, 3), boundless=True, fill_value=0)
            out[r.pid] = (int(w[1, 1]), w)
    return lyr, out
jobs = [(t, l) for t in pts.tile.unique() for l in LAYERS]
with ThreadPoolExecutor(12) as ex:
    res = list(ex.map(work, jobs))
recs = {p: {} for p in pts.pid}
for lyr, out in res:
    for p, (c, w) in out.items():
        r = recs[p]
        if lyr == 'treecover2000':
            r['tc2000'] = c; r['tc2000_3x3_mean'] = float(w.mean())
        elif lyr == 'gain':
            r['gain'] = c; r['gain_3x3_any'] = int((w > 0).any())
        else:
            r['lossyear'] = c  # 0 = no loss, 1..24 = 2001..2024
            ly = w[w > 0]
            r['loss_3x3_n'] = int((w > 0).sum())
            r['loss_3x3_first'] = int(ly.min()) if ly.size else 0
            r['loss_3x3_le2020_n'] = int(((w > 0) & (w <= 20)).sum())
h = pd.DataFrame.from_dict(recs, orient='index').rename_axis('pid').reset_index()
h.to_csv(f'{D}/hansen_gfc.csv', index=False)
m = h.merge(pts[['pid', 'dataset', 'label']], on='pid')
m['loss_le2020'] = m.lossyear.between(1, 20); m['loss_after2020'] = m.lossyear > 20
print(m.groupby(['dataset', 'label'])[['tc2000', 'gain', 'loss_le2020', 'loss_after2020', 'loss_3x3_le2020_n']].mean().round(3))
