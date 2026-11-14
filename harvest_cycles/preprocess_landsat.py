#!/usr/bin/env python3
"""
Raw Landsat -> cloud-masked dated observations -> yearly NBR/NDVI series at a set of points
(step 1 of 2; step 2 is build_harvest_cycles_features.py).

Reads single pixels of Landsat 4/5/7/8 Collection 1 Level-1 scenes from Google's public bucket with HTTP range
requests (no account needed). The cost follows the number of scenes, that is of Landsat path/rows the points fall in
(about 215 scenes each), more than the number of points: about 2 minutes for 12 points in 6 path/rows (32 workers)
and 25 minutes for 2,264 points in 38 path/rows (64 workers) on a fast connection, plus about a minute to stream the
scene index once; much longer on a laptop. For large areas or production, use Earth Engine and Collection 2 instead
(README, "Before production").

Usage:
    python preprocess_landsat.py --points POINTS.csv --out DIR [--workers 32] [--hansen] [--index INDEX.csv]
        [--set name=value ...]

    POINTS.csv  one row per point: pid (unique id), lon, lat (WGS84 degrees). Other columns are ignored.
    DIR         output folder, one per set of points. Put it inside the git-ignored CROPIN-MIT-DATA/ folder.
                The run is resumable: after an interruption, run the same command again.

Steps (finished steps are skipped on a re-run; steps 5-6 always run, they take a minute or two):
  1. landsat_index.csv      scene index for the points' area: streams Google's full index once (about 1 GB);
                            --index reuses a file written by an earlier run that covers the same area
  2. pathrows.csv           nearest Landsat path/row centres of each point (pathrow_centres.csv)
  3. scenes_selected.csv    least cloudy scenes per path/row, year and season (scenes_all.csv: all candidates)
  4. cache/pixels/*.csv     top-of-atmosphere reflectance and quality flags at the points, one file per scene;
                            scenes that cannot be read are listed in failed_scenes.csv and retried on a re-run
  5. observations.csv.gz    NBR and NDVI per point and clear observation (sensors harmonised, same-pass duplicates
                            merged); series_<index>_<aggregation>.csv.gz (one value per point and year, columns =
                            years); series_obs_counts.csv.gz; sensor_offsets.csv
  6. hansen_gfc.csv         with --hansen: Hansen tree cover, gain and loss year at the points, for
                            build_harvest_cycles_features.py --calibrate-levels
DIR/settings.json records the settings, inputs and software versions. All settings are in config.py.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PreprocessSettings, with_overrides, write_run_record  # noqa: E402

log = logging.getLogger("harvest_cycles.preprocess")

SPACECRAFT = {"LT04": "LANDSAT_4", "LT05": "LANDSAT_5", "LE07": "LANDSAT_7", "LC08": "LANDSAT_8"}
BANDS = {"LT04": {"red": 3, "nir": 4, "swir2": 7}, "LT05": {"red": 3, "nir": 4, "swir2": 7},
         "LE07": {"red": 3, "nir": 4, "swir2": 7}, "LC08": {"red": 4, "nir": 5, "swir2": 7}}
PIXEL_COLUMNS = ["pid", "product_id", "qa", "clear", "red", "nir", "swir2", "red_dn", "nir_dn", "swir2_dn"]
# Settings that change what steps 1-4 download; a cache made with other values is not reused.
DOWNLOAD_SETTINGS = ["first_year", "last_year", "landsat_bucket", "sensors", "index_year_padding", "region_margin_degrees",
                     "max_daytime_row", "min_scenes_per_pathrow", "second_pathrow_max_km", "max_cloud_cover",
                     "scenes_per_season", "dry_months", "slc_off_date", "slc_off_penalty"]
_local = threading.local()


# ----------------------------------------------------------------------------------------------- helpers
def configure_gdal() -> None:
    """GDAL settings for single-pixel reads over HTTP (set before rasterio opens a file); values already set in the
    environment are kept."""
    for key, value in dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF,.tif",
                           GDAL_HTTP_MAX_RETRY="8", GDAL_HTTP_RETRY_DELAY="2", GDAL_CACHEMAX="64",
                           CPL_VSIL_CURL_CHUNK_SIZE="65536", VSI_CACHE="FALSE", GDAL_HTTP_TIMEOUT="60",
                           GDAL_HTTP_CONNECTTIMEOUT="20").items():
        os.environ.setdefault(key, value)


def session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


@contextmanager
def thread_pool(workers: int):
    """A thread pool that, on Ctrl+C or an error, drops its queued tasks instead of running them all first
    (the tasks already running still finish)."""
    ex = ThreadPoolExecutor(workers)
    try:
        yield ex
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()


def rasterio_io_error() -> type:
    """rasterio's read error (step 6), without importing rasterio when it is not installed."""
    try:
        from rasterio.errors import RasterioIOError
        return RasterioIOError
    except ImportError:
        return DownloadError


class DownloadError(Exception):
    """A download that failed after every retry (or was cut off)."""


class CoverageError(SystemExit):
    """Points outside every Landsat scene of the index."""


def http_get(url: str, params: dict | None = None, tries: int = 6, timeout: int = 60) -> requests.Response:
    """GET with retries and backoff for connection errors, timeouts (408), rate limits (429) and server errors (5xx).
    Other client errors (400, 401, 403, 404, ...) raise requests.HTTPError at once; DownloadError after the last try."""
    status = None
    for k in range(tries):
        if k:
            time.sleep(2 * k - 1)
        try:
            r = session().get(url, params=params, timeout=timeout)
        except requests.RequestException:
            if k == tries - 1:
                raise
            continue
        if r.status_code == 200:
            return r
        status = r.status_code
        if 400 <= status < 500 and status not in (408, 429):
            r.raise_for_status()
    raise DownloadError(f"GET failed after {tries} tries (last status {status}): {url}")


def write_csv_atomic(df: pd.DataFrame, path: Path, **kw) -> None:
    """Write to a temporary file, then rename, so an interrupted run never leaves a half-written file. A .gz name
    gets gzip with no time stamp or file name in its header, so identical data always gives identical bytes."""
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".gz":
        with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
                io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
            df.to_csv(text, **kw)
    else:
        df.to_csv(tmp, **kw)
    os.replace(tmp, path)


def read_points(path: Path) -> pd.DataFrame:
    pts = pd.read_csv(path, dtype={"pid": str})
    missing = {"pid", "lon", "lat"} - set(pts.columns)
    if missing:
        raise SystemExit(f"{path}: missing column(s) {sorted(missing)}; need pid, lon, lat")
    if pts.empty:
        raise SystemExit(f"{path}: no points")
    if not (pd.api.types.is_numeric_dtype(pts["lon"]) and pd.api.types.is_numeric_dtype(pts["lat"])):
        raise SystemExit(f"{path}: lon and lat must be numbers (decimal degrees)")
    if pts["pid"].isna().any() or pts["pid"].duplicated().any():
        raise SystemExit(f"{path}: pid must be present and unique")
    bad = ~pts["lon"].between(-180, 180) | ~pts["lat"].between(-90, 90)
    if bad.any():
        raise SystemExit(f"{path}: {int(bad.sum())} points with invalid lon/lat")
    return pts[["pid", "lon", "lat"]].reset_index(drop=True)


def check_manifest(out: Path, pts: pd.DataFrame, s: PreprocessSettings) -> dict:
    """Refuse to reuse a folder whose cached downloads were made for other points or other download settings.
    Returns the manifest of this run, written by write_manifest once the points are known to be covered."""
    text = "\n".join(f"{p},{x!r},{y!r}" for p, x, y in sorted(zip(pts.pid, pts.lon, pts.lat)))
    manifest = dict(points_sha256=hashlib.sha256(text.encode()).hexdigest(), n_points=len(pts),
                    download_settings={k: asdict(s)[k] for k in DOWNLOAD_SETTINGS})
    manifest = json.loads(json.dumps(manifest))
    path = out / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != manifest:
            why = [f"{k} ({old.get('download_settings', {}).get(k)!r} there, {v!r} now)"
                   for k, v in manifest["download_settings"].items() if old.get("download_settings", {}).get(k) != v]
            if old.get("points_sha256") != manifest["points_sha256"]:
                why.insert(0, f"the points ({old.get('n_points')} there, {len(pts)} now, or the same number with other "
                              "ids or coordinates)")
            raise SystemExit(f"{out} was started for other inputs: {'; '.join(why)}. Use a new --out folder (or "
                             "empty this one).")
    return manifest


def write_manifest(out: Path, manifest: dict) -> None:
    path = out / "manifest.json"
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=2) + "\n")


# ----------------------------------------------------------------------------------------------- 1. index
def build_index(pts: pd.DataFrame, s: PreprocessSettings, path: Path) -> None:
    """Stream Google's Landsat scene index and keep the scenes of the wanted sensors and years whose footprint
    intersects the points' bounding box plus a margin."""
    m = s.region_margin_degrees
    lat0, lat1 = pts.lat.min() - m, pts.lat.max() + m
    lon0, lon1 = pts.lon.min() - m, pts.lon.max() + m
    y0, y1 = s.first_year - s.index_year_padding, s.last_year + s.index_year_padding
    spacecraft = {SPACECRAFT[x] for x in s.sensors}
    url = f"https://storage.googleapis.com/{s.landsat_bucket}/index.csv.gz"
    log.info("step 1: streaming the scene index %s (about 1 GB, once)", url)
    tmp = path.with_name(path.name + ".tmp")
    try:
        _stream_index(url, tmp, spacecraft, (y0, y1), (lat0, lat1, lon0, lon1))
    except (requests.RequestException, OSError, EOFError, urllib3.exceptions.HTTPError) as e:
        raise DownloadError(f"reading the scene index failed ({e!r})") from e
    os.replace(tmp, path)


def _stream_index(url, tmp, spacecraft, years, box):
    """Write the index rows of the wanted spacecraft, years (y0, y1) and box (lat0, lat1, lon0, lon1) to tmp."""
    (y0, y1), (lat0, lat1, lon0, lon1) = years, box
    n = kept = 0
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        rows = csv.reader(io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw), encoding="utf-8", newline=""))
        header = next(rows)
        ix = {k: i for i, k in enumerate(header)}
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for row in rows:
                n += 1
                try:
                    if row[ix["SPACECRAFT_ID"]] not in spacecraft:
                        continue
                    if not y0 <= int(row[ix["DATE_ACQUIRED"]][:4]) <= y1:
                        continue
                    north, south = float(row[ix["NORTH_LAT"]]), float(row[ix["SOUTH_LAT"]])
                    west, east = float(row[ix["WEST_LON"]]), float(row[ix["EAST_LON"]])
                except (ValueError, IndexError):
                    continue
                if south > lat1 or north < lat0 or west > lon1 or east < lon0:
                    continue
                w.writerow(row)
                kept += 1
    log.info("step 1: kept %d of %d index rows", kept, n)


# ----------------------------------------------------------------------------------------------- 2. path/rows
def assign_pathrows(pts: pd.DataFrame, index_path: Path, s: PreprocessSettings, out: Path) -> pd.DataFrame:
    """Each point's nearest and second-nearest path/row centre (median footprint centre of the L1TP scenes)."""
    ix = pd.read_csv(index_path)
    ix = ix[ix.WRS_ROW <= s.max_daytime_row]
    ix["clat"] = (ix.NORTH_LAT + ix.SOUTH_LAT) / 2
    ix["clon"] = (ix.WEST_LON + ix.EAST_LON) / 2
    pr = (ix[ix.DATA_TYPE == "L1TP"].groupby(["WRS_PATH", "WRS_ROW"])
          .agg(clat=("clat", "median"), clon=("clon", "median"), n=("SCENE_ID", "size"),
               north=("NORTH_LAT", "median"), south=("SOUTH_LAT", "median"),
               west=("WEST_LON", "median"), east=("EAST_LON", "median")).reset_index())
    pr = pr[pr.n >= s.min_scenes_per_pathrow].reset_index(drop=True)
    if pr.empty:
        raise CoverageError(f"no Landsat path/row of {index_path} covers these points: check that lon and lat are not "
                            "swapped")
    lon, lat = pts.lon.to_numpy()[:, None], pts.lat.to_numpy()[:, None]
    covered = ((lat <= pr.north.to_numpy()) & (lat >= pr.south.to_numpy())
               & (lon >= pr.west.to_numpy()) & (lon <= pr.east.to_numpy())).any(axis=1)
    if not covered.all():
        outside = pts.pid[~covered].tolist()
        raise CoverageError(f"{len(outside)} point(s) outside every Landsat scene of {index_path}, e.g. "
                            f"{', '.join(outside[:5])}: check that lon and lat are not swapped (and, with --index, "
                            "that the index covers these points)")
    radius = 6371.0
    lat = np.deg2rad(pts.lat.to_numpy())[:, None]
    dx = np.deg2rad(pts.lon.to_numpy()[:, None] - pr.clon.to_numpy()[None, :]) * np.cos(lat) * radius
    dy = np.deg2rad(pts.lat.to_numpy()[:, None] - pr.clat.to_numpy()[None, :]) * radius
    dist = np.hypot(dx, dy)
    order = np.argsort(dist, axis=1)
    name = [f"{p:03d}{r:03d}" for p, r in zip(pr.WRS_PATH, pr.WRS_ROW)]
    rows = np.arange(len(pts))
    out_pr = pd.DataFrame({"pid": pts.pid, "pr1": [name[o[0]] for o in order], "d1_km": dist[rows, order[:, 0]]})
    if len(pr) > 1:
        out_pr["pr2"] = [name[o[1]] for o in order]
        out_pr["d2_km"] = dist[rows, order[:, 1]]
    else:
        out_pr["pr2"], out_pr["d2_km"] = None, np.inf
    pr.to_csv(out / "pathrow_centres.csv", index=False)
    return out_pr


# ----------------------------------------------------------------------------------------------- 3. scenes
def list_prefix(bucket: str, prefix: str) -> list[str]:
    api = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    out, token = [], None
    while True:
        params = {"prefix": prefix, "delimiter": "/", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        d = http_get(api, params).json()
        out += d.get("prefixes", [])
        token = d.get("nextPageToken")
        if not token:
            return out


def mtl_text(bucket: str, prefix: str) -> str:
    product = prefix.rstrip("/").split("/")[-1]
    return http_get(f"https://storage.googleapis.com/{bucket}/{prefix}{product}_MTL.txt").text


def mtl_value(text: str, key: str) -> float:
    m = re.search(rf"\b{key} = ([\-\d\.E\+]+)", text)
    return float(m.group(1)) if m else np.nan


def select_scenes(pathrows: pd.DataFrame, index_path: Path, s: PreprocessSettings, workers: int):
    """List the Collection 1 L1TP scenes of every path/row from the bucket itself (the index has gaps), attach
    cloud cover (index, else the scene's metadata file) and keep the scenes_per_season least cloudy scenes of each
    season per path/row and year (Landsat 7 after the SLC failure penalised)."""
    prs = sorted(set(pathrows.pr1) | set(pathrows.loc[pathrows.d2_km < s.second_pathrow_max_km, "pr2"]))
    log.info("step 3: listing scenes for %d path/rows", len(prs))
    jobs = [f"{sensor}/01/{p[:3]}/{p[3:]}/" for p in prs for sensor in s.sensors]
    with thread_pool(16) as ex:
        listed = list(ex.map(lambda prefix: list_prefix(s.landsat_bucket, prefix), jobs))
    pattern = re.compile(r"(L[TEC]0\d)_(L1\w\w)_(\d{3})(\d{3})_(\d{8})_(\d{8})_01_(T1|T2|RT)/$")
    rows = []
    for prefixes in listed:
        for prefix in prefixes:
            m = pattern.search(prefix)
            if m:
                sensor, level, path, row, acq, proc, tier = m.groups()
                rows.append(dict(product_id=prefix.rstrip("/").split("/")[-1], prefix=prefix, sensor=sensor, level=level,
                                 pr=path + row, acq=acq, proc=proc, tier=tier))
    sc = pd.DataFrame(rows, columns=["product_id", "prefix", "sensor", "level", "pr", "acq", "proc", "tier"])
    sc = sc[sc.level == "L1TP"]
    # one product per acquisition: tier T1 > T2 > RT, then the latest processing
    sc["tier_rank"] = sc.tier.map({"T1": 0, "T2": 1, "RT": 2})
    sc = sc.sort_values(["sensor", "pr", "acq", "tier_rank", "proc"], ascending=[True, True, True, True, False])
    sc = sc.drop_duplicates(["sensor", "pr", "acq"])
    sc["date"] = pd.to_datetime(sc.acq)
    sc["year"], sc["month"] = sc.date.dt.year, sc.date.dt.month
    sc = sc[(sc.year >= s.first_year) & (sc.year <= s.last_year)]
    ix = pd.read_csv(index_path, usecols=["PRODUCT_ID", "CLOUD_COVER"])
    sc = sc.merge(ix.rename(columns={"PRODUCT_ID": "product_id", "CLOUD_COVER": "cc"}), on="product_id", how="left")
    ix["k"], sc["k"] = ix.PRODUCT_ID.str[:25], sc.product_id.str[:25]      # same acquisition, other processing
    sc["cc"] = sc.cc.fillna(sc.k.map(ix.drop_duplicates("k").set_index("k").CLOUD_COVER))
    missing = sc[sc.cc.isna()]
    log.info("step 3: %d scenes listed, %d without cloud cover in the index (read from their metadata)", len(sc), len(missing))

    def cloud_from_mtl(prefix):
        try:
            return mtl_value(mtl_text(s.landsat_bucket, prefix), "CLOUD_COVER")
        except requests.HTTPError:   # no metadata file (403/404): the scene is dropped below. Network errors stop the
            return np.nan            # step instead, so a selection that depends on them is never saved.

    with thread_pool(workers) as ex:
        sc.loc[missing.index, "cc"] = list(ex.map(cloud_from_mtl, missing.prefix))
    n_unknown = int(sc.cc.isna().sum())
    if n_unknown:
        log.warning("step 3: %d scenes dropped: no cloud cover in the index or in their metadata", n_unknown)
    sc = sc[sc.cc.notna() & (sc.cc >= 0) & (sc.cc < s.max_cloud_cover)]
    slc_off = (sc.sensor == "LE07") & (sc.date >= s.slc_off_date)
    sc["rank_score"] = sc.cc + np.where(slc_off, s.slc_off_penalty, 0)
    sc["season"] = np.where(sc.month.isin(s.dry_months), "dry", "wet")
    k = s.scenes_per_season
    picks = []
    for _, grp in sc.groupby(["pr", "year"]):
        chosen = []
        for season in ["dry", "wet"]:
            chosen += list(grp[grp.season == season].nsmallest(k, "rank_score").index)
        if len(chosen) < 2 * k:
            chosen += list(grp.drop(chosen).nsmallest(2 * k - len(chosen), "rank_score").index)
        picks += chosen
    return sc, sc.loc[picks].copy()


# ----------------------------------------------------------------------------------------------- 4. pixels
def decode_clear(qa: int, sensor: str) -> bool:
    """Collection 1 BQA: not fill, not dropped pixel (TM/ETM+) or terrain occlusion (OLI), radiometric saturation code
    at most 1, no cloud, and low cloud, shadow and snow (and cirrus for OLI) confidence. These rules are fixed here
    rather than in config.py: they are the standard clear-sky reading of the QA band."""
    fill, dropped, sat, cloud = qa & 1, (qa >> 1) & 1, (qa >> 2) & 3, (qa >> 4) & 1
    cloud_conf, shadow_conf, snow_conf = (qa >> 5) & 3, (qa >> 7) & 3, (qa >> 9) & 3
    cirrus_conf = (qa >> 11) & 3 if sensor == "LC08" else 0
    return bool(fill == 0 and dropped == 0 and sat <= 1 and cloud == 0 and cloud_conf <= 1 and shadow_conf <= 1
                and snow_conf <= 1 and cirrus_conf <= 1)


def extract_scene(scene, points: np.ndarray, bucket: str, out_dir: Path) -> tuple[int, int]:
    """Quality flags at the candidate points, and top-of-atmosphere reflectance where clear; one CSV per scene."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window
    base = f"https://storage.googleapis.com/{bucket}/{scene.prefix}{scene.product_id}"
    mtl = mtl_text(bucket, scene.prefix)
    sin_sun = np.sin(np.deg2rad(mtl_value(mtl, "SUN_ELEVATION")))
    coef = {b: (mtl_value(mtl, f"REFLECTANCE_MULT_BAND_{n}"), mtl_value(mtl, f"REFLECTANCE_ADD_BAND_{n}"))
            for b, n in BANDS[scene.sensor].items()}
    with rasterio.open(f"/vsicurl/{base}_BQA.TIF") as q:
        x, y = Transformer.from_crs("EPSG:4326", q.crs, always_xy=True).transform(points[:, 1].astype(float),
                                                                                  points[:, 2].astype(float))
        rr, cc = (np.asarray(a) for a in rasterio.transform.rowcol(q.transform, x, y))
        inside = np.where((rr >= 0) & (rr < q.height) & (cc >= 0) & (cc < q.width))[0]
        inside = inside[np.lexsort((cc[inside] // 256, rr[inside] // 256))]       # read block by block
        qa = {i: int(q.read(1, window=Window(int(cc[i]), int(rr[i]), 1, 1))[0, 0]) for i in inside}
    clear = [i for i in inside if decode_clear(qa[i], scene.sensor)]
    dn = {}
    for b, n in BANDS[scene.sensor].items():
        if clear:
            with rasterio.open(f"/vsicurl/{base}_B{n}.TIF") as src:
                for i in clear:
                    dn[(b, i)] = int(src.read(1, window=Window(int(cc[i]), int(rr[i]), 1, 1))[0, 0])
    recs = []
    for i in inside:
        rec = dict(pid=points[i, 0], product_id=scene.product_id, qa=qa[i], clear=int(("nir", i) in dn))
        if ("nir", i) in dn:
            for b in BANDS[scene.sensor]:
                mult, add = coef[b]
                rec[b] = np.nan if dn[(b, i)] == 0 else (mult * dn[(b, i)] + add) / sin_sun
                rec[b + "_dn"] = dn[(b, i)]
        recs.append(rec)
    write_csv_atomic(pd.DataFrame(recs, columns=PIXEL_COLUMNS), out_dir / f"{scene.product_id}.csv", index=False)
    return len(inside), len(clear)


def extract_all(pts: pd.DataFrame, pathrows: pd.DataFrame, scenes: pd.DataFrame, s: PreprocessSettings,
                pixel_dir: Path, workers: int, failed_path: Path) -> int:
    """Run extract_scene for every selected scene not yet in the cache. Scenes that fail after the retries are listed
    in failed_path (removed when none fails); a re-run tries them again. Returns the number of failed scenes."""
    pixel_dir.mkdir(parents=True, exist_ok=True)
    p = pts.merge(pathrows, on="pid")
    candidates = {pr: p.loc[(p.pr1 == pr) | ((p.pr2 == pr) & (p.d2_km < s.second_pathrow_max_km)), ["pid", "lon", "lat"]]
                  .to_numpy(dtype=object) for pr in scenes.pr.unique()}
    done = {f.name[:-4] for f in pixel_dir.glob("*.csv")}
    todo = scenes[~scenes.product_id.isin(done)]
    log.info("step 4: %d scenes, %d already extracted, %d to do (%d workers)", len(scenes), len(done), len(todo), workers)
    t0, n_done, n_obs, failed = time.time(), 0, 0, []
    with thread_pool(workers) as ex:
        futures = {ex.submit(extract_scene, r, candidates[r.pr], s.landsat_bucket, pixel_dir): r.product_id
                   for r in todo.itertuples()}
        for f in as_completed(futures):
            try:
                n_obs += f.result()[0]
                n_done += 1
            except Exception as e:   # noqa: BLE001 - logged and listed; a re-run retries the scene
                failed.append(dict(product_id=futures[f], error=repr(e)))
                log.warning("scene %s failed: %r", futures[f], e)
            if (n_done + len(failed)) % 250 == 0:
                log.info("step 4: %d/%d scenes, %d point observations, %.1f min, %d failed",
                         n_done + len(failed), len(todo), n_obs, (time.time() - t0) / 60, len(failed))
    log.info("step 4: finished %d scenes in %.1f min, %d failed", n_done, (time.time() - t0) / 60, len(failed))
    if failed:
        write_csv_atomic(pd.DataFrame(failed).sort_values("product_id"), failed_path, index=False)
    else:
        failed_path.unlink(missing_ok=True)
    return len(failed)


# ----------------------------------------------------------------------------------------------- 5. observations
def load_pixels(pixel_dir: Path, scenes: pd.DataFrame, pids) -> pd.DataFrame:
    files = sorted(glob.glob(str(pixel_dir / "*.csv")))
    obs = pd.concat((pd.read_csv(f, dtype={"pid": str}) for f in files), ignore_index=True)
    obs = obs[obs.product_id.isin(scenes.product_id) & obs.pid.isin(pids)]
    obs = obs.merge(scenes[["product_id", "sensor", "date"]], on="product_id", how="left")
    obs["date"] = pd.to_datetime(obs["date"])
    obs["year"], obs["month"] = obs.date.dt.year, obs.date.dt.month
    return obs.sort_values(["pid", "date", "product_id"]).reset_index(drop=True)


def add_indices(obs: pd.DataFrame, s: PreprocessSettings) -> pd.DataFrame:
    """NBR and NDVI from top-of-atmosphere reflectance; impossible or non-vegetated-land values set to NaN."""
    b = obs[["red", "nir", "swir2"]]
    ok = (obs["clear"] == 1) & (b > 0).all(axis=1) & (b < s.max_reflectance).all(axis=1)
    obs["nbr"] = np.where(ok, (obs.nir - obs.swir2) / (obs.nir + obs.swir2), np.nan)
    obs["ndvi"] = np.where(ok, (obs.nir - obs.red) / (obs.nir + obs.red), np.nan)
    bad = (obs.nir < s.min_nir) | (obs.nbr < s.min_nbr) | (obs.ndvi < s.min_ndvi)
    obs.loc[bad, ["nbr", "ndvi"]] = np.nan
    return obs


def estimate_offset(obs: pd.DataFrame, index: str, sensor: str, years: tuple, s: PreprocessSettings):
    """Median (sensor - Landsat 7) difference for the same point seen by both within offset_pair_max_days.
    Returns (offset or None if too few pairs, number of pairs, bootstrap 95% CI)."""
    o = obs[obs[index].notna() & obs.year.between(*years)]
    m = o[o.sensor == sensor][["pid", "date", index]].merge(o[o.sensor == "LE07"][["pid", "date", index]], on="pid",
                                                           suffixes=("_s", "_r"))
    m = m[(m.date_s - m.date_r).abs().dt.days <= s.offset_pair_max_days]
    d = (m[f"{index}_s"] - m[f"{index}_r"]).to_numpy()
    if len(d) < s.offset_min_pairs:
        return None, len(d), (np.nan, np.nan)
    rng = np.random.default_rng(0)
    boots = [np.median(rng.choice(d, len(d))) for _ in range(500)]
    return float(np.median(d)), len(d), tuple(np.percentile(boots, [2.5, 97.5]))


def harmonise(obs: pd.DataFrame, s: PreprocessSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put Landsat 8 OLI and Landsat 4/5 TM on the Landsat 7 ETM+ scale. Applies the offsets from config.py, or the
    offsets measured on these points when estimate_offsets is set (and there are enough pairs)."""
    report = []
    for index in ["nbr", "ndvi"]:
        oli, n_oli, ci_oli = estimate_offset(obs, index, "LC08", s.offset_years_oli, s)
        tm, n_tm, ci_tm = estimate_offset(obs, index, "LT05", s.offset_years_tm, s)
        fixed_oli, fixed_tm = getattr(s, f"{index}_oli_minus_etm"), getattr(s, f"{index}_tm_minus_etm")
        use_oli = oli if s.estimate_offsets and oli is not None else fixed_oli
        use_tm = tm if s.estimate_offsets and tm is not None else fixed_tm
        if s.estimate_offsets:
            for name, measured, n in [("Landsat 8", oli, n_oli), ("Landsat 4/5", tm, n_tm)]:
                if measured is None:
                    log.warning("%s %s offset: only %d same-point pairs; using the config.py value", name, index, n)
                else:
                    log.info("%s %s offset measured on these points: %+.4f (%d pairs)", name, index, measured, n)
        for sensor, off in {"LC08": use_oli, "LT05": use_tm, "LT04": use_tm}.items():
            obs.loc[obs.sensor == sensor, index] = obs.loc[obs.sensor == sensor, index] - off
        report.append(dict(index=index, oli_minus_etm_applied=use_oli, tm_minus_etm_applied=use_tm,
                           oli_minus_etm_measured=oli, n_pairs_oli=n_oli, oli_ci_low=ci_oli[0], oli_ci_high=ci_oli[1],
                           tm_minus_etm_measured=tm, n_pairs_tm=n_tm, tm_ci_low=ci_tm[0], tm_ci_high=ci_tm[1]))
    return obs, pd.DataFrame(report)


def dedupe_same_pass(obs: pd.DataFrame) -> pd.DataFrame:
    """A point where two rows of one path overlap is seen twice per pass: keep one observation per point, sensor and
    date (mean of the two)."""
    keys = ["pid", "sensor", "date"]
    agg = {c: "mean" for c in ["nbr", "ndvi"]}
    agg.update({c: "first" for c in obs.columns if c not in agg and c not in keys})
    return obs.groupby(keys, as_index=False).agg(agg)


def yearly_series(obs: pd.DataFrame, index: str, aggregation: str, pids, s: PreprocessSettings) -> pd.DataFrame:
    """One value per point and year (columns = years, NaN = no clear observation)."""
    years = np.arange(s.first_year, s.last_year + 1)
    o = obs[obs[index].notna() & obs.year.between(s.first_year, s.last_year)]
    if aggregation == "dry_median":
        o = o[o.month.isin(s.dry_months)]
    elif aggregation == "wet_median":
        o = o[~o.month.isin(s.dry_months)]
    elif aggregation not in ("median", "max"):
        raise ValueError(f"unknown aggregation {aggregation!r}")
    g = o.groupby(["pid", "year"])[index]
    series = (g.max() if aggregation == "max" else g.median()).unstack("year").reindex(columns=years)
    series = series.reindex(pids)
    series.index.name = "pid"
    return series


def build_observations(pts: pd.DataFrame, scenes: pd.DataFrame, pixel_dir: Path, s: PreprocessSettings, out: Path) -> None:
    (out / "settings.json").unlink(missing_ok=True)    # the run record is rewritten when the run finishes
    wanted = {f"series_{index}_{aggregation}.csv.gz" for index, aggregation in s.series} | {"series_obs_counts.csv.gz"}
    for stale in out.glob("series_*.csv.gz"):            # series of an earlier run with other settings
        if stale.name not in wanted:
            stale.unlink()
    obs = load_pixels(pixel_dir, scenes, set(pts.pid))
    log.info("step 5: %d point observations, %d clear, %d points", len(obs), int(obs["clear"].sum()), obs.pid.nunique())
    obs = add_indices(obs, s)
    obs, offsets = harmonise(obs, s)
    offsets.to_csv(out / "sensor_offsets.csv", index=False)
    n0 = int(obs.nbr.notna().sum())
    obs = dedupe_same_pass(obs)
    log.info("step 5: %d same-pass duplicates merged", n0 - int(obs.nbr.notna().sum()))
    years = np.arange(s.first_year, s.last_year + 1)
    counts = (obs[obs.nbr.notna()].groupby(["pid", "year"]).size().unstack("year")
              .reindex(index=pts.pid, columns=years).fillna(0).astype(int))
    write_csv_atomic(counts, out / "series_obs_counts.csv.gz")
    for index, aggregation in s.series:
        series = yearly_series(obs, index, aggregation, pts.pid, s)
        write_csv_atomic(series, out / f"series_{index}_{aggregation}.csv.gz")
        log.info("step 5: series_%s_%s: %.1f%% of point-years without a clear observation",
                 index, aggregation, 100 * float(series.isna().mean().mean()))
    clean = obs[obs.nbr.notna() | obs.ndvi.notna()]
    write_csv_atomic(clean[["pid", "product_id", "sensor", "date", "year", "month", "nbr", "ndvi"]],
                     out / "observations.csv.gz", index=False)
    log.info("step 5: observations.csv.gz: %d clear observations; median %.0f per point and year",
             len(clean), float(np.median(counts.to_numpy())))
    empty = counts.index[counts.sum(axis=1) == 0].tolist()
    if empty:
        log.warning("step 5: %d points have no clear observation in any year (e.g. %s)", len(empty), ", ".join(empty[:5]))


# ----------------------------------------------------------------------------------------------- 6. Hansen
def sample_hansen(pts: pd.DataFrame, s: PreprocessSettings, path: Path) -> None:
    """Hansen Global Forest Change at each point: centre pixel and its 3x3 neighbourhood (tree cover 2000, gain
    2000-2012, loss year 1..24 = 2001..2024; losses after last_year are ignored by the features)."""
    import rasterio
    from rasterio.windows import Window

    def tile(lon, lat):
        top, left = math.ceil(lat / 10) * 10, math.floor(lon / 10) * 10
        return f"{abs(top):02d}{'N' if top >= 0 else 'S'}_{abs(left):03d}{'E' if left >= 0 else 'W'}"

    p = pts.assign(tile=[tile(a, b) for a, b in zip(pts.lon, pts.lat)])
    last_loss = s.last_year - 2000

    def work(job):
        """3x3 windows of one layer around the points of one tile. Neighbours outside the tile (points on the tile's
        edge pixels) are returned as coordinates to read from the next tile."""
        t, layer = job
        with rasterio.open("/vsicurl/" + s.hansen_url.format(layer=layer, tile=t)) as src:
            out, outside = {}, []
            for r in p[p.tile == t].itertuples():
                row, col = src.index(r.lon, r.lat)
                w = src.read(1, window=Window(col - 1, row - 1, 3, 3), boundless=True, fill_value=0)
                for i in range(3):
                    for j in range(3):
                        rr, cc = row - 1 + i, col - 1 + j
                        if not (0 <= rr < src.height and 0 <= cc < src.width):
                            outside.append((r.pid, layer, i, j, *src.xy(rr, cc)))
                out[r.pid] = w
        return layer, out, outside

    def read_pixels(job):
        """Single pixels of one layer of one tile: [(pid, layer, i, j, lon, lat), ...] -> values."""
        (t, layer), pixels = job
        with rasterio.open("/vsicurl/" + s.hansen_url.format(layer=layer, tile=t)) as src:
            return [(pid, lyr, i, j, int(src.read(1, window=Window(c, r, 1, 1))[0, 0]))
                    for pid, lyr, i, j, x, y in pixels for r, c in [src.index(x, y)]]

    jobs = [(t, layer) for t in p.tile.unique() for layer in ["treecover2000", "gain", "lossyear"]]
    windows, edge = {}, {}
    with thread_pool(12) as ex:
        for layer, values, outside in ex.map(work, jobs):
            for pid, w in values.items():
                windows[(pid, layer)] = w
            for px in outside:
                edge.setdefault((tile(px[4], px[5]), layer), []).append(px)
        for filled in ex.map(read_pixels, edge.items()):       # neighbours in the next tile
            for pid, layer, i, j, v in filled:
                windows[(pid, layer)][i, j] = v
    if edge:
        log.info("step 6: %d neighbour pixels read from an adjacent tile", sum(len(v) for v in edge.values()))
    recs = {pid: {} for pid in p.pid}
    for (pid, layer), w in windows.items():
        r, c = recs[pid], int(w[1, 1])
        if layer == "treecover2000":
            r["tc2000"], r["tc2000_3x3_mean"] = c, float(w.mean())
        elif layer == "gain":
            r["gain"], r["gain_3x3_any"] = c, int((w > 0).any())
        else:
            loss = w[w > 0]
            r["lossyear"], r["loss_3x3_n"] = c, int((w > 0).sum())
            r["loss_3x3_first"] = int(loss.min()) if loss.size else 0
            r["loss_3x3_le2020_n"] = int(((w > 0) & (w <= last_loss)).sum())   # losses up to last_year;
            # the column name keeps the default 2020
    h = pd.DataFrame.from_dict(recs, orient="index").rename_axis("pid").reset_index()
    write_csv_atomic(h, path, index=False)
    log.info("step 6: Hansen values for %d points", len(h))


# ----------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=Path, required=True, help="CSV with pid, lon, lat")
    p.add_argument("--out", type=Path, required=True, help="output folder (inside the git-ignored CROPIN-MIT-DATA/)")
    p.add_argument("--workers", type=int, default=32, help="parallel downloads (default 32)")
    p.add_argument("--index", type=Path, help="reuse a landsat_index.csv from an earlier run covering the same area")
    p.add_argument("--hansen", action="store_true", help="also sample Hansen Global Forest Change (step 6)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                   help="override a PreprocessSettings value from config.py for this run (repeatable)")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be at least 1")
    if args.index is not None and not args.index.exists():
        p.error(f"--index {args.index} does not exist (leave out --index to build the index in --out)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_gdal()
    try:
        with_overrides(PreprocessSettings(), args.overrides)
    except ValueError as e:
        p.error(str(e))
    try:
        return run(args, argv)
    except KeyboardInterrupt:
        log.warning("interrupted; the finished steps and scenes are kept: run the same command again to resume")
        return 130
    except (requests.RequestException, DownloadError, rasterio_io_error()) as e:
        log.error("download failed after retries: %s. Finished steps are kept: run the same command again.", e)
        return 1


def run(args: argparse.Namespace, argv: list[str]) -> int:
    s = with_overrides(PreprocessSettings(), args.overrides)
    pts = read_points(args.points)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = check_manifest(out, pts, s)
    log.info("%d points -> %s", len(pts), out)

    index_path = args.index or out / "landsat_index.csv"
    pathrows_path, selected_path = out / "pathrows.csv", out / "scenes_selected.csv"
    if not index_path.exists() and not (pathrows_path.exists() and selected_path.exists()):
        build_index(pts, s, index_path)
    if pathrows_path.exists():
        pathrows = pd.read_csv(pathrows_path, dtype={"pid": str, "pr1": str, "pr2": str})
    else:
        try:
            pathrows = assign_pathrows(pts, index_path, s, out)
        except CoverageError:
            if args.index is None:            # an index built for wrong coordinates must not be reused
                index_path.unlink(missing_ok=True)
            raise
        write_csv_atomic(pathrows, pathrows_path, index=False)
    write_manifest(out, manifest)
    log.info("step 2: %d path/rows (%d points also use a second one)", pathrows.pr1.nunique(),
             int((pathrows.d2_km < s.second_pathrow_max_km).sum()))
    if selected_path.exists():
        scenes = pd.read_csv(selected_path, dtype={"pr": str, "acq": str, "proc": str})
    else:
        all_scenes, scenes = select_scenes(pathrows, index_path, s, args.workers)
        write_csv_atomic(all_scenes, out / "scenes_all.csv", index=False)
        write_csv_atomic(scenes, selected_path, index=False)
    log.info("step 3: %d scenes selected (median %.0f per path/row and year)", len(scenes),
             float(scenes.groupby(["pr", "year"]).size().median()))
    pixel_dir = out / "cache" / "pixels"
    failed = extract_all(pts, pathrows, scenes, s, pixel_dir, args.workers, out / "failed_scenes.csv")
    if failed > s.max_failed_scene_share * len(scenes):
        log.error("%d of %d scenes could not be read (listed in failed_scenes.csv): too many to go on, probably a "
                  "network problem. Run the same command again to retry them.", failed, len(scenes))
        return 1
    if failed:
        log.warning("%d of %d scenes could not be read after retries (listed in failed_scenes.csv); going on without "
                    "them. Run the same command again later to retry them.", failed, len(scenes))
    build_observations(pts, scenes, pixel_dir, s, out)
    if args.hansen:
        sample_hansen(pts, s, out / "hansen_gfc.csv")
    record = write_run_record(out, s, dict(points=args.points, index=index_path if index_path.exists() else None),
                              dict(failed_scenes=failed), script=Path(__file__).name, command=argv)
    log.info("done; settings recorded in %s", record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
