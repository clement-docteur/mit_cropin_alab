"""Pick 30 MapBiomas points for a check by eye in historical high-resolution imagery (Google Earth).

The check tests two things at once: whether SDPT's species labels (pine / eucalyptus) are right at our points, and
whether the harvest-cycle detector finds the clear-cuts that are visible in the imagery.

Strata (random within each, fixed seed):
  8 plantations in SDPT pine        (4 caught, 4 missed by the main model, H-all random forest)
  8 plantations in SDPT eucalyptus  (4 caught, 4 missed)
  8 plantations outside SDPT        (4 caught, 4 missed)
  5 natural-forest reference points inside an SDPT plantation that the main model calls plantation (all of them)
  1 natural-forest control outside SDPT that passes "forest every year since 1985"
Points are then shuffled and named E01..E30, so the order gives nothing away.

Needs results from notebook_predictions.py and species_analysis.py. Writes species_split/eye_check/:
  eye_check_points.csv   what the reviewer sees: id, coordinates, links (no answers)
  eye_check_key.csv      what our data says at each point (open only after recording)
  eye_check_points.kml   for Google Earth Pro: a pin and the ~30 m Landsat pixel at each point
  eye_check_data.json    everything above plus the yearly NBR series, for the recording page
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "species_split" / "results"
OUT = REPO / "species_split" / "eye_check"
OUT.mkdir(exist_ok=True)
SEED = 7

v = pd.read_csv(RES / "val_points_species.csv")
ev = pd.read_csv(RES / "cut_events_val.csv")
nbr = pd.read_csv(REPO / "public_data/harvest_cycles/series_nbr_median.csv.gz", index_col=0)
hist = pd.read_csv(REPO / "public_data/harvest_cycles/val_label_history.csv").set_index("pid")

rng = np.random.default_rng(SEED)
pl = v[v.label == 1]
picks = []
for species, name in [("pine", "SDPT pine plantation"), ("eucalyptus", "SDPT eucalyptus plantation"),
                      ("none", "plantation outside SDPT")]:
    for pred, call in [(1, "caught"), (0, "missed")]:
        pool = pl[(pl.sdpt_species == species) & (pl.hall_rf_pred == pred)]
        picks.append(pool.sample(4, random_state=int(rng.integers(1e9))).assign(stratum=f"{name}, {call} by the model"))
fo = v[v.label == 0]
picks.append(fo[(fo.sdpt_species != "none") & (fo.hall_rf_pred == 1)].assign(
    stratum="natural forest (reference) inside an SDPT plantation, called plantation by the model"))
picks.append(fo[(fo.sdpt_species == "none") & (fo.forest_since_1985 == 1) & (fo.hall_rf_pred == 0)].sample(
    1, random_state=int(rng.integers(1e9))).assign(stratum="natural forest control"))
s = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)
s.insert(0, "id", [f"E{i + 1:02d}" for i in range(len(s))])
assert len(s) == 30 and s.pid.is_unique

cols = [f"CLASS_{y}" for y in range(1985, 2023)]
s["ref_years_plantation"] = [",".join(str(y) for y, c in zip(range(1985, 2023), hist.loc[p, cols]) if c == 9) for p in s.pid]
s["detected_cut_years"] = [",".join(str(int(y)) for y in sorted(ev[ev.pid == p].cut_year)) for p in s.pid]


def earth(lat, lon):
    return f"https://earth.google.com/web/@{lat:.6f},{lon:.6f},0a,500d,35y,0h,0t,0r"


def maps(lat, lon):
    return f"https://www.google.com/maps/@{lat:.6f},{lon:.6f},250m/data=!3m1!1e3"


s["google_earth"] = [earth(a, b) for a, b in zip(s.lat, s.lon)]
s["google_maps"] = [maps(a, b) for a, b in zip(s.lat, s.lon)]
s[["id", "lat", "lon", "state", "google_earth", "google_maps"]].to_csv(OUT / "eye_check_points.csv", index=False)
key = s[["id", "pid", "stratum", "sdpt_species", "sdpt_planted_year", "label", "group", "hall_rf_pred", "hall_rf_score",
         "n_cuts", "detected_cut_years", "abrupt_start", "forest_since_1985", "ref_years_plantation"]].rename(
    columns={"label": "mapbiomas_reference (1 = plantation)", "group": "mapbiomas_10m_map", "hall_rf_pred": "model_calls_plantation"})
key.to_csv(OUT / "eye_check_key.csv", index=False)


# KML: a pin and a ~30 m square (the Landsat pixel, centred on the point) for every point
def square(lat, lon, half_m):
    dlat, dlon = half_m / 111_320, half_m / (111_320 * np.cos(np.radians(lat)))
    c = [(lon - dlon, lat - dlat), (lon + dlon, lat - dlat), (lon + dlon, lat + dlat), (lon - dlon, lat + dlat), (lon - dlon, lat - dlat)]
    return " ".join(f"{x:.7f},{y:.7f},0" for x, y in c)


marks = []
for r in s.itertuples():
    marks.append(f"""  <Placemark><name>{r.id}</name><styleUrl>#pin</styleUrl>
    <Point><coordinates>{r.lon:.7f},{r.lat:.7f},0</coordinates></Point></Placemark>
  <Placemark><name>{r.id} pixel</name><styleUrl>#px</styleUrl>
    <Polygon><outerBoundaryIs><LinearRing><coordinates>{square(r.lat, r.lon, 15)}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>""")
kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Harvest-cycle eye check (30 points)</name>
  <Style id="pin"><IconStyle><scale>0.8</scale></IconStyle><LabelStyle><scale>1.1</scale></LabelStyle></Style>
  <Style id="px"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>
{chr(10).join(marks)}
</Document></kml>
"""
(OUT / "eye_check_points.kml").write_text(kml)

# data for the recording page
data = []
for r in s.itertuples():
    e = ev[ev.pid == r.pid]
    series = nbr.loc[r.pid].to_numpy(float)
    data.append(dict(
        id=r.id, pid=r.pid, lat=round(r.lat, 6), lon=round(r.lon, 6), state=r.state, stratum=r.stratum,
        sdpt=r.sdpt_species, sdpt_planted=None if pd.isna(r.sdpt_planted_year) else int(r.sdpt_planted_year),
        ref=int(r.label), ref_years=r.ref_years_plantation, map10m=r.group, model=int(r.hall_rf_pred),
        score=round(float(r.hall_rf_score), 3),
        cuts=[dict(cut=int(c.cut_year), low=int(c.low_year), drop=round(float(c.drop), 2), cycle=bool(c.cycle)) for c in e.itertuples()],
        nbr=[None if np.isnan(x) else round(float(x), 3) for x in series]))
(OUT / "eye_check_data.json").write_text(json.dumps(dict(years=list(range(1985, 2021)), canopy=0.584, open=0.405, points=data)))
print(s[["id", "pid", "state", "stratum", "sdpt_species", "n_cuts", "detected_cut_years"]].to_string())
