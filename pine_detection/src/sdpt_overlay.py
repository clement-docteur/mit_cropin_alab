"""Overlay WRI's Spatial Database of Planted Trees (SDPT v2.0) on labelled points.

This reads SDPT polygons from Global Forest Watch's public vector tiles (no API key needed). Version
`v20231128` is SDPT v2.0; the newer `v20239998` tiles are empty over southern Brazil. For each point it
records the species of the polygon that contains it, and the species and distance of the nearest polygon
within --near metres.

Usage (from the repo root):
    python pine_detection/src/sdpt_overlay.py                       # public MapBiomas points
    python pine_detection/src/sdpt_overlay.py --points my.csv --out data/processed/x.csv
    python pine_detection/src/sdpt_overlay.py --bbox                # fetch every tile over PR/SC/RS

--points needs columns pid, lon, lat (other columns are kept). For Cropin QC polygons, run with --bbox
first. That fetches all tiles over the three states, so GFW never learns where the QC polygons are. The
overlay itself then runs from the local cache.

Tiles are cached in data/processed/sdpt_tiles/ (gitignored). The output goes to data/processed/
(gitignored). Requires: pandas, requests, shapely, mapbox-vector-tile.
"""
import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mapbox_vector_tile
import pandas as pd
import requests
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
TILE_URL = "https://tiles.globalforestwatch.org/gfw_planted_forests/{version}/dynamic/{z}/{x}/{y}.pbf"
VERSION = "v20231128"  # SDPT v2.0
Z = 10                 # ~40 km tiles, 4096 px extent -> ~10 m polygon detail
SOUTH_BBOX = (-58.0, -34.0, -48.0, -22.4)  # lon_min, lat_min, lon_max, lat_max (PR, SC, RS)
KEEP = ["sciname", "simplename", "simpletype", "reference", "imageryyear"]
M_PER_DEG = 111_320.0


def tile_of(lon, lat, z=Z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def tile_lonlat(px, py, tx, ty, z=Z, ext=4096):
    """Tile pixel (y up, as mapbox_vector_tile decodes by default) -> lon, lat."""
    n = 2 ** z
    X = (tx + px / ext) / n
    Y = (ty + (ext - py) / ext) / n
    return X * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * Y))))


def _convert(coords, tx, ty):
    if isinstance(coords[0], (int, float)):
        return tile_lonlat(coords[0], coords[1], tx, ty)
    return [_convert(c, tx, ty) for c in coords]


def fetch(tile, cache, version=VERSION, retries=3):
    x, y = tile
    f = cache / f"{version}_{Z}_{x}_{y}.pbf"
    if f.exists():
        return f
    for i in range(retries):
        try:
            r = requests.get(TILE_URL.format(version=version, z=Z, x=x, y=y), timeout=60)
            if r.status_code == 200:
                f.write_bytes(r.content)
                return f
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"tile {tile} failed after {retries} tries")


def load_tile(f, tx, ty):
    """Returns a list of (shapely geometry, properties) for one cached tile (empty if no polygons)."""
    raw = f.read_bytes()
    if not raw:
        return []
    out = []
    for layer in mapbox_vector_tile.decode(raw).values():
        for ft in layer["features"]:
            g = ft["geometry"]
            try:
                geom = shape({"type": g["type"], "coordinates": _convert(g["coordinates"], tx, ty)})
                if not geom.is_valid:
                    geom = geom.buffer(0)
            except Exception:
                continue
            out.append((geom, {k: ft["properties"].get(k) for k in KEEP}))
    return out


def bbox_tiles(bbox=SOUTH_BBOX):
    x0, y0 = tile_of(bbox[0], bbox[3])
    x1, y1 = tile_of(bbox[2], bbox[1])
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def overlay(points, cache, near_m=100.0):
    points = points.copy()
    points["tile"] = [tile_of(lo, la) for lo, la in zip(points.lon, points.lat)]
    rows = []
    for (tx, ty), grp in points.groupby("tile"):
        feats = load_tile(cache / f"{VERSION}_{Z}_{tx}_{ty}.pbf", tx, ty)
        tree = STRtree([g for g, _ in feats]) if feats else None
        for idx, r in grp.iterrows():
            rec = {"idx": idx, "sdpt_in": "none", "sdpt_near": "none", "sdpt_near_m": float("nan"),
                   "sdpt_n_polys_tile": len(feats), "sdpt_reference": None, "sdpt_imageryyear": None}
            if tree is not None:
                p = Point(r.lon, r.lat)
                hits = tree.query(p, predicate="intersects")
                if len(hits):
                    props = feats[hits[0]][1]
                    rec.update(sdpt_in=props["sciname"] or "Unknown", sdpt_reference=props["reference"],
                               sdpt_imageryyear=props["imageryyear"])
                # nearest polygon (distance in metres, small-angle approximation)
                j = tree.nearest(p)
                g, props = feats[j]
                d = g.distance(p) * M_PER_DEG * math.sqrt(
                    (math.cos(math.radians(r.lat)) ** 2 + 1) / 2)  # crude mean of lon/lat scale
                if d <= near_m:
                    rec.update(sdpt_near=props["sciname"] or "Unknown", sdpt_near_m=round(d, 1))
            rows.append(rec)
    res = pd.DataFrame(rows).set_index("idx")
    return points.drop(columns="tile").join(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=str(ROOT / "public_data/harvest_cycles/points.csv"))
    ap.add_argument("--dataset", default="val", help="keep rows with this `dataset` value ('' for all)")
    ap.add_argument("--out", default=str(ROOT / "data/processed/sdpt_overlay_mapbiomas.csv"))
    ap.add_argument("--cache", default=str(ROOT / "data/processed/sdpt_tiles"))
    ap.add_argument("--near", type=float, default=100.0, help="max distance (m) for the nearest polygon")
    ap.add_argument("--bbox", action="store_true", help="fetch every tile over PR/SC/RS, then stop")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)

    if a.bbox:
        tiles = bbox_tiles()
    else:
        pts = pd.read_csv(a.points)
        if a.dataset and "dataset" in pts:
            pts = pts[pts["dataset"] == a.dataset]
        tiles = sorted({tile_of(lo, la) for lo, la in zip(pts.lon, pts.lat)})

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        files = list(ex.map(lambda t: fetch(t, cache), tiles))
    mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"{len(tiles)} tiles ready ({mb:.0f} MB) in {time.time() - t0:.0f} s")
    if a.bbox:
        return

    res = overlay(pts, cache, a.near)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    print(f"wrote {len(res)} rows to {a.out}")


if __name__ == "__main__":
    main()
