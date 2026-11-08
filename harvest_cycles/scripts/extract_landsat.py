#!/usr/bin/env python3
"""Step 4: extract cloud-masked Landsat C1 TOA reflectance (red, NIR, SWIR2) at the points for the selected scenes.
Reads single pixels from tiled GeoTIFFs in gs://gcp-public-data-landsat via HTTP range requests (resumable:
finished scenes are skipped). BQA 'clear' = not fill/dropped, saturation <= 1 band group, no cloud bit, and
low cloud / shadow / snow (/ cirrus for OLI) confidence.
Usage: extract_landsat.py <data_dir> <workers> [--pilot PR YEAR]"""
import os, sys, re, time, threading, traceback
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.TIF',
                  GDAL_HTTP_MAX_RETRY='8', GDAL_HTTP_RETRY_DELAY='2', GDAL_CACHEMAX='64',
                  CPL_VSIL_CURL_CHUNK_SIZE='65536', VSI_CACHE='FALSE', GDAL_HTTP_TIMEOUT='60',
                  GDAL_HTTP_CONNECTTIMEOUT='20')
import numpy as np, pandas as pd, requests, rasterio
from rasterio.windows import Window
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor, as_completed

D = sys.argv[1]; WORKERS = int(sys.argv[2])
PILOT = sys.argv[4:6] if len(sys.argv) > 3 and sys.argv[3] == '--pilot' else None
BASE = 'https://storage.googleapis.com/gcp-public-data-landsat/'
BANDS = {'LT04': {'red': 3, 'nir': 4, 'swir2': 7}, 'LT05': {'red': 3, 'nir': 4, 'swir2': 7},
         'LE07': {'red': 3, 'nir': 4, 'swir2': 7}, 'LC08': {'red': 4, 'nir': 5, 'swir2': 7}}
pts = pd.read_csv(f'{D}/points.csv', dtype={'pr1': str, 'pr2': str})
sc = pd.read_csv(f'{D}/scenes_selected.csv', dtype={'pr': str})
if PILOT:
    sc = sc[(sc.pr == PILOT[0]) & (sc.year == int(PILOT[1]))]
# candidate points per path/row
cand = {}
for p in sc.pr.unique():
    m = (pts.pr1 == p) | ((pts.pr2 == p) & (pts.d2_km < 105))
    cand[p] = pts.loc[m, ['pid', 'lon', 'lat']].to_numpy()
outdir = f'{D}/landsat_parts' + ('_pilot' if PILOT else '')
os.makedirs(outdir, exist_ok=True)
done = {f[:-4] for f in os.listdir(outdir) if f.endswith('.csv')}
todo = sc[~sc.product_id.isin(done)]
print(f'scenes total {len(sc)}, done {len(done)}, todo {len(todo)}', flush=True)
tls = threading.local()

def sess():
    if not hasattr(tls, 's'):
        tls.s = requests.Session()
    return tls.s

def get_mtl(prefix, pid):
    url = f'{BASE}{prefix}{pid}_MTL.txt'
    for k in range(6):
        try:
            r = sess().get(url, timeout=60)
            if r.status_code == 200:
                return r.text
        except Exception:
            pass
        time.sleep(1 + 2 * k)
    raise RuntimeError('MTL fail ' + pid)

def mtl_val(t, key):
    m = re.search(rf'\b{key} = ([\-\d\.E\+]+)', t)
    return float(m.group(1)) if m else np.nan

def decode_clear(qa, sensor):
    fill = qa & 1; drop = (qa >> 1) & 1; sat = (qa >> 2) & 3; cloud = (qa >> 4) & 1
    cconf = (qa >> 5) & 3; sconf = (qa >> 7) & 3; snow = (qa >> 9) & 3
    cirr = (qa >> 11) & 3 if sensor == 'LC08' else 0
    return (fill == 0) & (drop == 0) & (sat <= 1) & (cloud == 0) & (cconf <= 1) & (sconf <= 1) & (snow <= 1) & (cirr <= 1)

def process(row):
    pid, prefix, sensor = row.product_id, row.prefix, row.sensor
    t0 = time.time(); nreq = 1
    mtl = get_mtl(prefix, pid)
    se = mtl_val(mtl, 'SUN_ELEVATION'); sin_se = np.sin(np.deg2rad(se))
    coef = {b: (mtl_val(mtl, f'REFLECTANCE_MULT_BAND_{n}'), mtl_val(mtl, f'REFLECTANCE_ADD_BAND_{n}'))
            for b, n in BANDS[sensor].items()}
    P = cand[row.pr]
    recs = []
    with rasterio.Env():
        with rasterio.open(f'/vsicurl/{BASE}{prefix}{pid}_BQA.TIF') as q:
            tr = Transformer.from_crs('EPSG:4326', q.crs, always_xy=True)
            x, y = tr.transform(P[:, 1].astype(float), P[:, 2].astype(float))
            rr, cc = rasterio.transform.rowcol(q.transform, x, y)
            rr = np.asarray(rr); cc = np.asarray(cc)
            inside = (rr >= 0) & (rr < q.height) & (cc >= 0) & (cc < q.width)
            idx = np.where(inside)[0]
            # sort by tile to reuse blocks
            idx = idx[np.lexsort((cc[idx] // 256, rr[idx] // 256))]
            qa = {}
            for i in idx:
                qa[i] = int(q.read(1, window=Window(int(cc[i]), int(rr[i]), 1, 1))[0, 0]); nreq += 1
        clear = [i for i in idx if decode_clear(qa[i], sensor)]
        vals = {}
        if clear:
            for b, n in BANDS[sensor].items():
                with rasterio.open(f'/vsicurl/{BASE}{prefix}{pid}_B{n}.TIF') as src:
                    for i in clear:
                        vals[(b, i)] = int(src.read(1, window=Window(int(cc[i]), int(rr[i]), 1, 1))[0, 0]); nreq += 1
    for i in idx:
        rec = dict(pid=P[i, 0], product_id=pid, qa=qa[i], clear=int(i in vals.get('_c', set()) or (('nir', i) in vals)))
        if ('nir', i) in vals:
            for b in BANDS[sensor]:
                dn = vals[(b, i)]
                m, a = coef[b]
                rec[b] = np.nan if dn == 0 else (m * dn + a) / sin_se
                rec[b + '_dn'] = dn
        recs.append(rec)
    df = pd.DataFrame(recs, columns=['pid', 'product_id', 'qa', 'clear', 'red', 'nir', 'swir2', 'red_dn', 'nir_dn', 'swir2_dn'])
    df.to_csv(f'{outdir}/{pid}.csv.tmp', index=False)
    os.replace(f'{outdir}/{pid}.csv.tmp', f'{outdir}/{pid}.csv')
    return pid, len(idx), len(clear), nreq, time.time() - t0

t0 = time.time(); n_done = 0; n_req = 0; n_pts = 0; errs = 0
with ThreadPoolExecutor(WORKERS) as ex:
    futs = {ex.submit(process, r): r.product_id for r in todo.itertuples()}
    for f in as_completed(futs):
        try:
            pid, ni, nc, nr, dt = f.result()
            n_done += 1; n_req += nr; n_pts += ni
        except Exception as e:
            errs += 1
            print('ERR', futs[f], repr(e)[:300], flush=True)
        if n_done % 50 == 0 or PILOT:
            el = time.time() - t0
            print(f'{n_done}/{len(todo)} scenes, {n_pts} pt-obs, {n_req} req, {n_req/el:.0f} req/s, '
                  f'elapsed {el/60:.1f} min, errors {errs}', flush=True)
print('finished', n_done, 'errors', errs, f'{(time.time()-t0)/60:.1f} min', flush=True)
