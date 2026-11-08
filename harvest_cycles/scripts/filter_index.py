#!/usr/bin/env python3
"""Step 1: stream the gcp-public-data-landsat index (~770 MB gz) and keep Landsat 4/5/7/8 scenes over
southern Brazil, 1984-2021. Usage: python filter_index.py <out.csv>"""
import csv, gzip, io, sys, requests
URL = "https://storage.googleapis.com/gcp-public-data-landsat/index.csv.gz"
out = sys.argv[1]
LAT = (-34.2, -22.3); LON = (-58.2, -47.6)
keep = 0; n = 0
with requests.get(URL, stream=True, timeout=600) as r:
    r.raise_for_status()
    r.raw.decode_content = False
    gz = gzip.GzipFile(fileobj=r.raw)
    txt = io.TextIOWrapper(gz, encoding="utf-8", newline="")
    rd = csv.reader(txt)
    header = next(rd)
    ix = {k: i for i, k in enumerate(header)}
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header)
        for row in rd:
            n += 1
            try:
                sc = row[ix["SPACECRAFT_ID"]]
                if sc not in ("LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"):
                    continue
                yr = int(row[ix["DATE_ACQUIRED"]][:4])
                if yr < 1984 or yr > 2021:
                    continue
                nl, sl = float(row[ix["NORTH_LAT"]]), float(row[ix["SOUTH_LAT"]])
                wl, el = float(row[ix["WEST_LON"]]), float(row[ix["EAST_LON"]])
            except (ValueError, IndexError):
                continue
            if sl > LAT[1] or nl < LAT[0] or wl > LON[1] or el < LON[0]:
                continue
            w.writerow(row); keep += 1
print("rows", n, "kept", keep)
