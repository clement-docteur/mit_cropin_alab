"""Rebuild the 2,290-point southern MapBiomas reference subset from the raw 85k file.

Input  : data/raw/mapbiomas_validation/mapbiomas_85k_points_validation.shp (local only, 209 MB .dbf)
Output : data/processed/mapbiomas_ref_south.gpkg  (and .csv without geometry)

Rule (CONTEXT.md §2b):
  - SIGLA in {PR, SC, RS}                      (Paraná, Santa Catarina, Rio Grande do Sul)
  - CLASS_2020 in {3, 9}                        (3 = forest formation, 9 = forest plantation)
  - CLASS_2016 == CLASS_2020 == CLASS_2022      (stable class across the period)
Fields: id = TARGETID, state, biome, cls2016/2020/2022, agree2020 = COUNT_2020,
        edge2020 = BORDA_2020, label, high_conf = (agree2020 == 3 and edge2020 == 0).

Coordinates come from the geometry: the raw LON/LAT columns are swapped (LON holds latitude).

Usage (from the repo root):
    python src/build_mapbiomas_ref_south.py [--check-zip PATH]
--check-zip defaults to data/gee_assets/mapbiomas_ref_south.zip, the asset uploaded to Earth Engine.
"""
import argparse
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/mapbiomas_validation/mapbiomas_85k_points_validation.shp"
OUT = ROOT / "data/processed/mapbiomas_ref_south"
COLS = ["TARGETID", "SIGLA", "BIOMA_250K", "LON", "LAT",
        "CLASS_2016", "CLASS_2020", "CLASS_2022", "COUNT_2020", "BORDA_2020"]


def build(raw_path=RAW):
    pts = gpd.read_file(raw_path, columns=COLS)

    # LON/LAT are swapped in the raw table; confirm it, then rely on the geometry only.
    assert (pts["LON"] - pts.geometry.y).abs().max() < 1e-6, "expected LON column to hold latitude"

    keep = (
        pts["SIGLA"].isin(["PR", "SC", "RS"])
        & pts["CLASS_2020"].isin([3, 9])
        & (pts["CLASS_2016"] == pts["CLASS_2020"])
        & (pts["CLASS_2022"] == pts["CLASS_2020"])
    )
    s = pts.loc[keep].copy()
    out = gpd.GeoDataFrame({
        "id": s["TARGETID"].astype(int),
        "state": s["SIGLA"],
        "biome": s["BIOMA_250K"],
        "cls2016": s["CLASS_2016"].astype(int),
        "cls2020": s["CLASS_2020"].astype(int),
        "cls2022": s["CLASS_2022"].astype(int),
        "agree2020": s["COUNT_2020"].astype(int),
        "edge2020": s["BORDA_2020"].astype(int),
        "label": s["CLASS_2020"].map({3: "forest", 9: "plantation"}),
        "lon": s.geometry.x,
        "lat": s.geometry.y,
    }, geometry=s.geometry, crs=pts.crs).reset_index(drop=True)
    out["high_conf"] = ((out["agree2020"] == 3) & (out["edge2020"] == 0)).astype(int)
    return out


def check(out, zip_path):
    assert len(out) == 2290, f"rows: {len(out)} != 2290"
    counts = out["label"].value_counts().to_dict()
    assert counts == {"forest": 1762, "plantation": 528}, f"labels: {counts}"
    assert out["high_conf"].sum() == 1448, f"high_conf: {out['high_conf'].sum()} != 1448"
    assert out["id"].is_unique

    ref = gpd.read_file(f"zip://{zip_path}")
    assert set(out["id"]) == set(ref["id"]), (
        f"ID mismatch: {len(set(out['id']) - set(ref['id']))} only in rebuild, "
        f"{len(set(ref['id']) - set(out['id']))} only in GEE asset")
    # Field-by-field agreement with the uploaded asset
    m = out.merge(ref.drop(columns="geometry"), on="id", suffixes=("", "_ref"))
    for c in ["cls2016", "cls2020", "cls2022", "agree2020", "edge2020", "label", "high_conf", "state"]:
        if f"{c}_ref" in m:
            assert (m[c] == m[f"{c}_ref"]).all(), f"field {c} differs from GEE asset"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check-zip", default=ROOT / "data/gee_assets/mapbiomas_ref_south.zip")
    args = ap.parse_args()

    out = build()
    check(out, args.check_zip)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(OUT.with_suffix(".gpkg"), driver="GPKG")
    out.drop(columns="geometry").to_csv(OUT.with_suffix(".csv"), index=False)

    print(f"OK: {len(out)} points written to {OUT.with_suffix('.gpkg').relative_to(ROOT)}")
    print(out["label"].value_counts().to_string())
    print(out["state"].value_counts().to_string())
    print(f"high_conf: {out['high_conf'].sum()}")
    print("All checks passed (row count, labels, high_conf, exact ID match with the GEE asset).")


if __name__ == "__main__":
    main()
