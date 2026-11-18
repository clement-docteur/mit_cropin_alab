"""Score the blind map check from gee/06.

Inputs (both local-only, in data/, never committed):
  data/gee_exports/map_check_key_<AREA>.csv   from the gee/06 export task (pid, dis, p_sag, p_six, lon, lat)
  data/processed/map_check_labels.csv         your labels: columns area, pid, label
                                              label is plantation / forest / unsure
Each area has 75 blind points, 25 per group:
  dis = 1, "winter adds":    only the 6-feature model (B) says plantation. B is right if you saw plantation.
  dis = 2, "winter removes": only SAG (A) says plantation. B is right if you saw forest.
  dis = 0, "agree":          both models give the same answer. A control: how often are both right?
                             Split by whether both say plantation or both say not.
Output is aggregate counts only.
Run:  .venv/bin/python src/score_map_check.py
"""
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_analysis import DATA, ROOT  # noqa: E402

# the cutoffs the map used, read from gee/06 so they can't drift
cfg = json.loads(re.search(r"var CONFIG = (\{.*?\});", (ROOT / "gee/06_map_view_winter_check.js").read_text(), re.S).group(1))
cut_a, cut_b = cfg["models"]["sag"]["cutoff"], cfg["models"]["six"]["cutoff"]

keys = []
for f in sorted(glob.glob(str(DATA / "gee_exports/map_check_key_*.csv"))):
    keys.append(pd.read_csv(f).assign(area=Path(f).stem.rsplit("_", 1)[1]))
assert keys, "no key files: run the map_check_key_<AREA> task in gee/06 and copy it to data/gee_exports/"
key = pd.concat(keys)
key["pid"] = key.pid.astype(int)

lab = pd.read_csv(DATA / "processed/map_check_labels.csv")
lab.columns = [c.strip().lower() for c in lab.columns]
lab = lab.dropna(subset=["label"])                 # rows not filled in yet are skipped
assert len(lab), "no labels filled in yet in data/processed/map_check_labels.csv"
lab["label"] = lab.label.astype(str)
lab["area"] = lab.area.str.strip().str.upper()
lab["label"] = lab.label.str.strip().str.lower()
assert lab.label.isin(["plantation", "forest", "unsure"]).all(), "labels must be plantation / forest / unsure"
assert not lab.duplicated(["area", "pid"]).any(), "an (area, pid) pair is labelled twice"
d = key.merge(lab, on=["area", "pid"], how="inner", validate="one_to_one")

d["group"] = d.dis.map({1: "winter adds", 2: "winter removes"})
both_plant = (d.dis == 0) & (d.p_sag >= cut_a) & (d.p_six >= cut_b)
d.loc[(d.dis == 0) & both_plant, "group"] = "agree: both say plantation"
d.loc[(d.dis == 0) & ~both_plant, "group"] = "agree: both say not plantation"
right_if = {"winter adds": "plantation", "winter removes": "forest",
            "agree: both say plantation": "plantation", "agree: both say not plantation": "forest"}
meaning = {"winter adds": "B right (you saw plantation)", "winter removes": "B right (you saw forest)",
           "agree: both say plantation": "both right (you saw plantation)",
           "agree: both say not plantation": "both right (you saw forest)"}

print(f"cutoffs: A {cut_a}, B {cut_b}")
for scope, g in [("ALL AREAS", d)] + [(a, x) for a, x in d.groupby("area")]:
    n_key = len(key) if scope == "ALL AREAS" else int((key.area == scope).sum())
    print(f"\n{scope}: labelled {len(g)} of {n_key}, unsure {int((g.label == 'unsure').sum())} "
          f"({(g.label == 'unsure').mean():.0%})")
    for grp in right_if:
        s = g[g.group == grp]
        sure = s[s.label != "unsure"]
        if len(sure) == 0:
            continue
        right = int((sure.label == right_if[grp]).sum())
        p = binomtest(right, len(sure), 0.5).pvalue
        print(f"  {grp:31s} {meaning[grp]}: {right} of {len(sure)} ({right / len(sure):.0%}), "
              f"p vs coin flip {p:.3f} | unsure {int((s.label == 'unsure').sum())} of {len(s)}")
print("\nRead: 'winter adds' and 'winter removes' well above 50% means the winter measurements fix more than they"
      "\nbreak. The 'agree' rows show how good the maps are where the models don't argue (should be high).")
