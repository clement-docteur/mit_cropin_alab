"""Extract plantation species from the Spatial Database of Planted Trees (SDPT v2.0, World Resources Institute)
at our points, and the planted area by species in southern Brazil.

SDPT v2 is published as one 7.7 GB zip that holds a zip that holds a 32 GB GeoPackage. This script never stores it:
it streams the file from Global Forest Watch's public bucket, decompresses both zip layers on the fly, and reads
the SQLite pages of the Brazil layer (`bra_plant_v2`) directly. It stops once that layer has been read, after
about 3.8 GB of download (a few minutes).

Writes to public_data/sdpt/:
- sdpt_polygons_at_points.geojson : every SDPT polygon whose bounding box is within ~300 m of one of our points,
                                    clipped to ~1 km boxes around those points
- sdpt_area_by_species.csv        : SDPT polygon count and area by species and state, within 30 km of the MapBiomas points

Run from the repository root:  python3 species_split/scripts/extract_sdpt.py
Needs numpy, pandas, scipy, shapely.
"""
import collections
import json
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "public_data" / "sdpt"
URL = "https://gfw2-data.s3.amazonaws.com/plantations/sdpt/sdpt_v2.zip"
OUTER_DATA_START = 867            # start of the deflated data of 'sdpt_v2_v20231128.gpkg.zip' inside sdpt_v2.zip
FIRST_PAGE, LAST_PAGE = 688_000, 922_000   # the Brazil layer's pages (its b-tree root is page 688,396; the next table's is 921,740)
PAGE = U = 4096
X, M = U - 35, ((U - 12) * 32 // 255) - 23

pts = pd.read_csv(REPO / "public_data" / "harvest_cycles" / "points.csv")
PX, PY = pts.lon.to_numpy(), pts.lat.to_numpy()
NEAR = 0.003                      # degrees (~300 m): keep polygons whose bounding box comes this close to a point
CLIP = 0.005                      # degrees (~500 m): saved polygons are clipped to boxes of this half-width around points
REGION = (PX.min() - 0.3, PX.max() + 0.3, PY.min() - 0.3, PY.max() + 0.3)


# ---------------- SQLite record parsing
def varint(b, i):
    v = 0
    for k in range(9):
        c = b[i + k]
        if k == 8:
            return (v << 8) | c, i + 9
        v = (v << 7) | (c & 0x7F)
        if c < 0x80:
            return v, i + k + 1


def serial_len(t):
    if t < 12:
        return [0, 1, 2, 3, 4, 6, 8, 8, 0, 0, 0, 0][t]
    return (t - 12) // 2 if t % 2 == 0 else (t - 13) // 2


def parse_record(payload):
    # [(serial type, value, offset)]; values past the end of a truncated payload are None
    hlen, i = varint(payload, 0)
    types = []
    while i < hlen:
        t, i = varint(payload, i)
        types.append(t)
    out, off = [], hlen
    for t in types:
        n = serial_len(t)
        v = None
        if off + n <= len(payload):
            raw = bytes(payload[off:off + n])
            if 1 <= t <= 6:
                v = int.from_bytes(raw, "big", signed=True)
            elif t == 7:
                v = struct.unpack(">d", raw)[0]
            elif t in (8, 9):
                v = t - 8
            elif t >= 13 and t % 2 == 1:
                v = raw.decode("utf-8", "replace")
            elif t >= 12:
                v = raw
        out.append((t, v, off))
        off += n
    return out


# ---------------- streaming pass
rows, master = [], []
ovf, wanted = {}, set()
recent = collections.OrderedDict()   # recent non-b-tree pages, for overflow chains that point backwards
missed = 0


def need(pg, cur):
    global missed
    while pg and pg not in ovf:
        if pg > cur:
            wanted.add(pg)
            return
        if pg not in recent:
            missed += 1
            return
        ovf[pg] = recent[pg]
        pg = struct.unpack(">I", ovf[pg][:4])[0]


def process_page(pno, page):
    off = 100 if pno == 1 else 0
    if pno in wanted:
        wanted.discard(pno)
        ovf[pno] = bytes(page)
        need(struct.unpack(">I", page[:4])[0], pno)
        return
    kind = page[off]
    if kind != 0x0D:                                    # not a table leaf: keep as a possible overflow page
        if kind not in (0x02, 0x05, 0x0A):
            recent[pno] = bytes(page)
            if len(recent) > 60_000:
                recent.popitem(last=False)
        return
    for c in range(struct.unpack(">H", page[off + 3:off + 5])[0]):
        cp = struct.unpack(">H", page[off + 8 + 2 * c:off + 10 + 2 * c])[0]
        P, i = varint(page, cp)
        rowid, i = varint(page, i)
        if P <= X:
            local = P
        else:
            K = M + ((P - M) % (U - 4))
            local = K if K <= X else M
        payload = bytes(page[i:i + local])
        first_ovf = struct.unpack(">I", page[i + local:i + local + 4])[0] if local < P else 0
        if b"GP" not in payload:
            continue
        rec = parse_record(payload)
        if len(rec) < 2 or rec[1][0] < 12:               # bra_plant_v2: OBJECTID, Shape, ...
            continue
        g = payload[rec[1][2]:rec[1][2] + 40]            # the geometry's header and envelope (the rest may overflow)
        if len(g) < 40 or g[:2] != b"GP" or not (g[3] >> 1) & 7:
            continue
        minx, maxx, miny, maxy = struct.unpack("<4d" if g[3] & 1 else ">4d", g[8:40])
        if maxx < REGION[0] or minx > REGION[1] or maxy < REGION[2] or miny > REGION[3]:
            continue
        near = ((PX >= minx - NEAR) & (PX <= maxx + NEAR) & (PY >= miny - NEAR) & (PY <= maxy + NEAR)).any()
        rows.append(dict(rowid=rowid, P=P, local=local, first_ovf=first_ovf, payload=payload, near=near,
                         env=(minx, maxx, miny, maxy)))
        if first_ovf:
            need(first_ovf, pno)


def stream():
    curl = subprocess.Popen(["curl", "-sS", "--retry", "5", "-r", f"{OUTER_DATA_START}-", URL], stdout=subprocess.PIPE)
    outer, inner = zlib.decompressobj(-15), None
    buf, gbuf, pno, done = bytearray(), bytearray(), 0, False
    while not done:
        chunk = curl.stdout.read(1 << 22)
        if not chunk:
            raise RuntimeError("download ended before the Brazil layer was read; run again")
        buf += outer.decompress(chunk)
        while inner is None and len(buf) >= 30:          # local headers of the inner zip: a folder, then the GeoPackage
            nl, el = struct.unpack("<HH", buf[26:30])
            name = bytes(buf[30:30 + nl])
            del buf[:30 + nl + el]
            if not name.endswith(b"/"):
                inner = zlib.decompressobj(-15)
        if inner is None:
            continue
        gbuf += inner.decompress(bytes(buf))
        buf.clear()
        n = len(gbuf) // PAGE
        mv = memoryview(gbuf)
        for k in range(n):
            pno += 1
            if pno == 1:
                assert bytes(mv[:16]) == b"SQLite format 3\x00"
            if FIRST_PAGE <= pno <= LAST_PAGE:
                process_page(pno, mv[k * PAGE:(k + 1) * PAGE])
            elif pno > LAST_PAGE:
                done = True
                break
        del mv
        del gbuf[:n * PAGE]
        if pno % 100_000 < n:
            print(f"page {pno:,} of {LAST_PAGE:,}, {len(rows):,} rows in the region", flush=True)
    curl.kill()


stream()

# ---------------- decode
COLS = ["OBJECTID", "Shape", "Shape_Length", "Shape_Area", "final_id", "iso3", "iso2", "country", "originalName",
        "originalCode", "finalCode", "vernacName", "sciName", "simpleName"] + [f"sciName{k}" for k in range(1, 14)] + [
        "simpleType", "leafStatus", "leafType", "woodType", "sizeCategory", "ownership", "reference", "imageryYear", "method",
        "removalFactorAGCBGCMgHaYr", "removalFactorAGCBGCstdevMgHaYr", "removalFactorAGCMgHaYr", "removalFactorAGCstdevMgHaYr",
        "removalFactorSource", "removalFactorStdevSource", "removalFactorNotes", "removalFactorSourceRegion", "plantedYear"]
KEEP = ["OBJECTID", "vernacName", "sciName", "originalName", "sizeCategory", "reference", "imageryYear", "plantedYear"]
near_feats, region = [], []
for r in rows:
    payload, todo, pg = r["payload"], r["P"] - r["local"], r["first_ovf"]
    parts = []
    while todo > 0 and pg in ovf:
        take = min(todo, U - 4)
        parts.append(ovf[pg][4:4 + take])
        todo -= take
        pg = struct.unpack(">I", ovf[pg][:4])[0]
    if todo > 0:
        missed += 1
        continue
    vals = [v for _, v, _ in parse_record(payload + b"".join(parts))]
    rec = dict(zip(COLS, vals))
    rec["OBJECTID"] = r["rowid"]
    if rec["iso3"] != "BRA":
        continue
    minx, maxx, miny, maxy = r["env"]
    region.append(dict(vernacName=rec["vernacName"], originalName=rec["originalName"], Shape_Area=rec["Shape_Area"],
                       cx=(minx + maxx) / 2, cy=(miny + maxy) / 2))
    if r["near"]:
        g = rec["Shape"]
        geom = shapely.from_wkb(g[8 + {1: 32, 2: 48, 3: 48, 4: 64}[(g[3] >> 1) & 7]:])
        # clip to ~1 km boxes around the points it is near, to keep the file small (point-in-polygon is unchanged)
        close = (PX >= minx - NEAR) & (PX <= maxx + NEAR) & (PY >= miny - NEAR) & (PY <= maxy + NEAR)
        boxes = shapely.union_all(shapely.box(PX[close] - CLIP, PY[close] - CLIP, PX[close] + CLIP, PY[close] + CLIP))
        geom = shapely.make_valid(geom).intersection(boxes)
        if geom.is_empty:
            continue
        near_feats.append({"type": "Feature", "properties": {k: rec[k] for k in KEEP},
                           "geometry": json.loads(shapely.to_geojson(geom))})
assert missed == 0, f"{missed} rows lost an overflow page"
print(f"Brazil rows in the region: {len(region):,}; polygons near a point: {len(near_feats):,}")

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "sdpt_polygons_at_points.geojson", "w") as f:
    json.dump({"type": "FeatureCollection", "features": near_feats}, f)

reg = pd.DataFrame(region)
reg["species"] = reg.vernacName.str.lower().where(reg.vernacName.isin(["Pine", "Eucalyptus", "Acacia"]), "other or unknown")
reg.loc[reg.originalName.str.contains("Clearing", na=False), "species"] = "young plantation or clearing"
vp = pts[pts.dataset == "val"]
scale = np.cos(np.radians(-27))
dist, idx = cKDTree(np.c_[vp.lon * scale, vp.lat]).query(np.c_[reg.cx * scale, reg.cy])
reg["state"] = vp.state.to_numpy()[idx]           # the state of the nearest MapBiomas point
reg = reg[dist * 111.32 <= 30]                    # within 30 km of a MapBiomas point
reg["area_ha"] = reg.Shape_Area * 111_320 ** 2 * np.cos(np.radians(reg.cy)) / 1e4
area = reg.groupby(["state", "species"]).agg(polygons=("area_ha", "size"), area_ha=("area_ha", "sum")).round(0).reset_index()
area.to_csv(OUT / "sdpt_area_by_species.csv", index=False)
print(area.pivot(index="species", columns="state", values="area_ha"))
