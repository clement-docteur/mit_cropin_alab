"""Is pine or eucalyptus over-represented among the plantations the harvest-cycle notebook gets wrong?

Species come from the Spatial Database of Planted Trees (SDPT v2.0, WRI; for Brazil: Transparent World's mapping of
large industrial plantations from 2013-14 imagery, Petersen et al. 2016), extracted by extract_sdpt.py. Model calls
come from notebook_predictions.py. Run both first, then:  python3 species_split/scripts/species_analysis.py
Prints every table (saved as results/species_analysis.txt) and writes results/misclassified_points.csv and
results/val_points_species.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import statsmodels.api as sm
from scipy import stats
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "species_split" / "results"
SDPT = REPO / "public_data" / "sdpt"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

pts = pd.read_csv(REPO / "public_data/harvest_cycles/points.csv")
gj = json.load(open(SDPT / "sdpt_polygons_at_points.geojson"))["features"]
poly = pd.DataFrame([f["properties"] for f in gj])
poly["geometry"] = [shapely.geometry.shape(f["geometry"]) for f in gj]
poly["Shape_Area"] = shapely.area(poly.geometry.values)   # clipped area: only used to pick the smallest polygon
pred = pd.read_csv(RESULTS / "model_predictions_val.csv")

# ---------------- 1. SDPT polygon at each point
SP = {"Pine": "pine", "Eucalyptus": "eucalyptus", "Acacia": "acacia"}
poly["species"] = poly.vernacName.map(SP).fillna("other/unknown")
poly.loc[poly.originalName.str.contains("Clearing", na=False), "species"] = "young/clearing"
geoms = poly.geometry.values
tree = shapely.STRtree(geoms)
P = shapely.points(pts.lon.values, pts.lat.values)
inside_pt, inside_poly = tree.query(P, predicate="intersects")
coslat = np.cos(np.radians(pts.lat.values))
rows = []
for i in range(len(pts)):
    hits = inside_poly[inside_pt == i]
    rec = dict(pid=pts.pid[i])
    if len(hits):
        j = hits[np.argmin(poly.Shape_Area.values[hits])]
        rec.update(sdpt_species=poly.species.iat[j], sdpt_planted_year=poly.plantedYear.iat[j], sdpt_id=int(poly.OBJECTID.iat[j]),
                   sdpt_dist_m=0.0)
    else:
        rec.update(sdpt_species="none", sdpt_planted_year=np.nan, sdpt_id=np.nan)
    # distance to the nearest SDPT polygon (approximate metres) and its species
    k = tree.nearest(P[i])
    if k is not None:
        dist_deg = shapely.distance(P[i], geoms[k])
        # convert with a local scale: degrees -> metres (mean of lon/lat scales is good enough at these distances)
        rec["nearest_dist_m"] = dist_deg * 111320 * (1 + coslat[i]) / 2
        rec["nearest_species"] = poly.species.iat[k]
    rows.append(rec)
sp = pd.DataFrame(rows)
# polygons were kept only within ~300 m of a point, so larger distances mean "more than ~300 m"
sp.loc[sp.nearest_dist_m > 300, ["nearest_species"]] = "none within 300 m"
sp["sdpt_within_30m"] = np.where(sp.sdpt_species != "none", sp.sdpt_species,
                                 np.where(sp.nearest_dist_m <= 30, sp.nearest_species, "none"))
sp = pts[["pid", "dataset", "lon", "lat", "state", "CD_MUN"]].merge(sp, on="pid")

V = pred.merge(sp[["pid", "sdpt_species", "sdpt_within_30m", "sdpt_planted_year", "nearest_dist_m"]], on="pid")
V["cluster"] = V.CD_MUN.astype(str)
ORDER = ["pine", "eucalyptus", "acacia", "young/clearing", "other/unknown", "none"]


def pct(x):
    return f"{100 * x:.0f}%"


print("=" * 100)
print("SDPT polygons near our points (clipped):", len(poly), poly.species.value_counts().to_dict())
print("\n1. Where are the MapBiomas points in SDPT (polygon containing the point, 2013-14 mapping)?")
t = pd.crosstab(V.sdpt_species, V.label.map({1: "plantation (ref)", 0: "natural forest (ref)"})).reindex(ORDER).fillna(0).astype(int)
t["plantation share"] = (t["plantation (ref)"] / t["plantation (ref)"].sum()).map(pct)
print(t)
t30 = pd.crosstab(V.sdpt_within_30m, V.label).reindex(ORDER).fillna(0).astype(int)
print("   (counting a polygon within 30 m too):", t30[1].to_dict(), "| forest:", t30[0].to_dict())

# ---------------- 2. sample mix vs regional plantation area
area = pd.read_csv(SDPT / "sdpt_area_by_species.csv").pivot(index="species", columns="state", values="area_ha")
area = area.rename(index={"other or unknown": "other/unknown", "young plantation or clearing": "young/clearing"}).reindex(ORDER[:-1]).fillna(0)
area["all"] = area.sum(axis=1)
print("\n2. Plantation area mapped by SDPT (2013-14) within 30 km of the MapBiomas points, thousand ha")
print((area / 1e3).round(0))
share_area = area.loc[["pine", "eucalyptus"], "all"] / area.loc[["pine", "eucalyptus"], "all"].sum()
pl = V[(V.label == 1) & V.sdpt_species.isin(["pine", "eucalyptus"])]
share_pts = pl.sdpt_species.value_counts(normalize=True)
print(f"   pine : eucalyptus by mapped area = {share_area['pine']:.0%} : {share_area['eucalyptus']:.0%};"
      f" among MapBiomas plantation points inside SDPT = {share_pts['pine']:.0%} : {share_pts['eucalyptus']:.0%}"
      f" (n = {len(pl)}; binomial p = {stats.binomtest((pl.sdpt_species == 'pine').sum(), len(pl), share_area['pine']).pvalue:.2g})")
for st in ["PR", "SC", "RS"]:
    a = area.loc[["pine", "eucalyptus"], st]
    s = pl[pl.state == st].sdpt_species.value_counts()
    print(f"   {st}: area pine {a['pine'] / a.sum():.0%} | points pine {s.get('pine', 0)}/{s.sum()}")

# ---------------- 3. model errors by species
MODELS = {"hall_rf": "H-all, random forest (main model)", "hall_lr": "H-all, logistic regression",
          "h4_rf": "H4, random forest", "full_rf": "36 yearly values, random forest", "hallv_rf": "H-all + variability, RF"}


def clustered_p(x, yv, cl):
    X = sm.add_constant(np.asarray(x, float)[:, None], has_constant="add")
    f = sm.GLM(np.asarray(yv, float), X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(np.asarray(cl))[0]})
    return np.asarray(f.pvalues)[1]


print("\n3. Plantations missed by each model (threshold: <= 5% of natural forest flagged), by SDPT species")
err_rows = []
for m, name in MODELS.items():
    PL = V[V.label == 1]
    r = {"model": name}
    for s in ORDER:
        g = PL[PL.sdpt_species == s]
        r[s] = f"{(g[m + '_pred'] == 0).sum()}/{len(g)} ({pct((g[m + '_pred'] == 0).mean()) if len(g) else '-'})"
    pe = PL[PL.sdpt_species.isin(["pine", "eucalyptus"])]
    miss = (pe[m + "_pred"] == 0).astype(int)
    is_pine = (pe.sdpt_species == "pine").astype(int)
    tab = pd.crosstab(is_pine, miss).reindex(index=[1, 0], columns=[1, 0]).fillna(0).to_numpy()
    r["Fisher p (pine vs euc)"] = f"{stats.fisher_exact(tab)[1]:.2g}"
    r["clustered p"] = f"{clustered_p(is_pine, miss, pe.cluster):.2g}"
    err_rows.append(r)
print(pd.DataFrame(err_rows).set_index("model").to_string())

m = "hall_rf"
PL = V[V.label == 1]
missed, caught = PL[PL[m + "_pred"] == 0], PL[PL[m + "_pred"] == 1]
print(f"\n   Main model: species mix of the {len(missed)} missed vs the {len(caught)} caught plantations")
mix = pd.DataFrame({"missed": missed.sdpt_species.value_counts(), "caught": caught.sdpt_species.value_counts()}).reindex(ORDER).fillna(0).astype(int)
mix["missed %"] = (mix.missed / mix.missed.sum()).map(pct)
mix["caught %"] = (mix.caught / mix.caught.sum()).map(pct)
print(mix)
# odds of being missed, pine vs eucalyptus, with 95% CI
pe = PL[PL.sdpt_species.isin(["pine", "eucalyptus"])]
a = ((pe.sdpt_species == "pine") & (pe[m + "_pred"] == 0)).sum(); b = ((pe.sdpt_species == "pine") & (pe[m + "_pred"] == 1)).sum()
c = ((pe.sdpt_species == "eucalyptus") & (pe[m + "_pred"] == 0)).sum(); e = ((pe.sdpt_species == "eucalyptus") & (pe[m + "_pred"] == 1)).sum()
orr = (a * e) / (b * c); se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / e)
print(f"   odds ratio of being missed, pine vs eucalyptus: {orr:.2f} ({np.exp(np.log(orr) - 1.96 * se):.2f}-{np.exp(np.log(orr) + 1.96 * se):.2f})")
# within SDPT-in-30m sensitivity
pe30 = PL[PL.sdpt_within_30m.isin(["pine", "eucalyptus"])]
for s in ["pine", "eucalyptus"]:
    g = pe30[pe30.sdpt_within_30m == s]
    print(f"   sensitivity (polygon within 30 m): {s} missed {(g[m + '_pred'] == 0).sum()}/{len(g)} ({pct((g[m + '_pred'] == 0).mean())})")

# ---------------- 4. forest points called plantation: are they in SDPT plantations?
FO = V[V.label == 0]
fp = FO[FO[m + "_pred"] == 1]
print(f"\n4. Natural-forest reference points the main model calls plantation: {len(fp)} of {len(FO)}")
print("   SDPT at those points:", fp.sdpt_species.value_counts().to_dict())
print("   SDPT at the other forest points:", FO[FO[m + '_pred'] == 0].sdpt_species.value_counts().to_dict())
inside = FO.sdpt_species != "none"
tab = pd.crosstab(inside, FO[m + "_pred"]).to_numpy()
print(f"   forest points inside an SDPT plantation: flagged {pct((fp.sdpt_species != 'none').mean())} vs not flagged "
      f"{pct((FO[FO[m + '_pred'] == 0].sdpt_species != 'none').mean())} (Fisher p = {stats.fisher_exact(tab)[1]:.2g})")
for mm in MODELS:
    f2 = FO[FO[mm + "_pred"] == 1]
    print(f"   {MODELS[mm]}: {len(f2)} forest flagged, {(f2.sdpt_species != 'none').sum()} inside SDPT "
          f"({(f2.sdpt_species == 'pine').sum()} pine, {(f2.sdpt_species == 'eucalyptus').sum()} eucalyptus)")
# MapBiomas reference history at forest points inside SDPT plantations
hist = pd.read_csv(REPO / "public_data/harvest_cycles/val_label_history.csv")
fo_in = FO[inside].merge(hist, on="pid")
ever9 = (fo_in[[f"CLASS_{y}" for y in range(1985, 2023)]] == 9).any(axis=1)
print(f"   forest points inside SDPT plantations: {inside.sum()}; any year labelled plantation by MapBiomas reference 1985-2022: {ever9.sum()}")

# ---------------- 5. harvest features by species: do pine and eucalyptus behave differently?
print("\n5. Harvest features by SDPT species, MapBiomas plantation points (natural forest outside SDPT for reference)")
PLf = PL.copy()
grp = PLf.groupby("sdpt_species")
FOo = FO[FO.sdpt_species == "none"]


def frow(g):
    return pd.Series({"points": len(g), "a cut": pct(g.has_cut.mean()), "a cycle": pct(g.has_cycle.mean()),
                      "2+ cycles": pct(g.two_plus_cycles.mean()), "mean cuts": f"{g.n_cuts.mean():.2f}",
                      "abrupt start": pct(g.abrupt_start.mean()), "forest since 1985": pct(g.forest_since_1985.mean()),
                      "cut 2010-20": pct(g.last_cut_year.between(2010, 2020).mean()),
                      "median NBR sd": f"{g.s_std.median():.3f}", "caught (H-all RF)": pct(g.hall_rf_pred.mean())})


ft = pd.concat({**{s: frow(grp.get_group(s)) for s in ORDER if s in grp.groups},
                "natural forest (outside SDPT)": frow(FOo)}, axis=1).T
print(ft.to_string())
pine, euc = PL[PL.sdpt_species == "pine"], PL[PL.sdpt_species == "eucalyptus"]
for col in ["has_cut", "has_cycle", "two_plus_cycles", "abrupt_start", "forest_since_1985"]:
    tab = [[pine[col].sum(), len(pine) - pine[col].sum()], [euc[col].sum(), len(euc) - euc[col].sum()]]
    both = pd.concat([pine, euc])
    print(f"   {col}: pine {pct(pine[col].mean())} vs eucalyptus {pct(euc[col].mean())}; Fisher p = {stats.fisher_exact(tab)[1]:.2g},"
          f" clustered p = {clustered_p((both.sdpt_species == 'pine').astype(int), both[col].astype(int), both.cluster):.2g}")
print(f"   n_cuts: Mann-Whitney p = {stats.mannwhitneyu(pine.n_cuts, euc.n_cuts).pvalue:.2g}; NBR sd p = {stats.mannwhitneyu(pine.s_std, euc.s_std).pvalue:.2g}")

# cut intervals, regrowth and size of cuts, pine vs eucalyptus
ev = pd.read_csv(RESULTS / "cut_events_val.csv").merge(V[["pid", "label", "sdpt_species"]], on="pid")
ev = ev[ev.label == 1].sort_values(["pid", "cut_year"])
ev["interval"] = ev.groupby("pid").cut_year.diff()
for col, what in [("interval", "years between consecutive cuts"), ("regrow_years", "years to regrow 80% (cycles)"),
                  ("drop", "size of the NBR fall")]:
    e_ = ev[ev.cycle] if col == "regrow_years" else ev
    a_, b_ = e_[e_.sdpt_species == "pine"][col].dropna(), e_[e_.sdpt_species == "eucalyptus"][col].dropna()
    c_ = e_[e_.sdpt_species == "none"][col].dropna()
    print(f"   {what}: pine median {a_.median():.2f} (n = {len(a_)}), eucalyptus {b_.median():.2f} (n = {len(b_)}), "
          f"not in SDPT {c_.median():.2f} (n = {len(c_)}); pine vs eucalyptus Mann-Whitney p = {stats.mannwhitneyu(a_, b_).pvalue:.2g}")
iv = ev.dropna(subset=["interval"])
for s_ in ["pine", "eucalyptus", "none"]:
    v_ = iv[iv.sdpt_species == s_].interval
    print(f"   {s_}: intervals <= 8 years {np.mean(v_ <= 8):.0%}, 9-14 years {np.mean((v_ > 8) & (v_ <= 14)):.0%}, > 14 years {np.mean(v_ > 14):.0%}")

# plantations inside vs outside SDPT (large industrial plantations vs the rest)
print("\n5b. Plantations inside an SDPT polygon (large industrial) vs outside it")
PL2 = PL.assign(in_sdpt=(PL.sdpt_species != "none").astype(int))
for mm in MODELS:
    miss_in = (PL2[PL2.in_sdpt == 1][mm + "_pred"] == 0)
    miss_out = (PL2[PL2.in_sdpt == 0][mm + "_pred"] == 0)
    tab = [[miss_out.sum(), len(miss_out) - miss_out.sum()], [miss_in.sum(), len(miss_in) - miss_in.sum()]]
    print(f"   {MODELS[mm]}: missed outside SDPT {miss_out.sum()}/{len(miss_out)} ({pct(miss_out.mean())}), inside "
          f"{miss_in.sum()}/{len(miss_in)} ({pct(miss_in.mean())}); Fisher p = {stats.fisher_exact(tab)[1]:.2g}, clustered p = "
          f"{clustered_p(1 - PL2.in_sdpt, (PL2[mm + '_pred'] == 0).astype(int), PL2.cluster):.2g}")
print("   by state, main model, share missed:", PL2.assign(miss=1 - PL2.hall_rf_pred).groupby(["state", "in_sdpt"]).miss.mean()
      .round(2).unstack().rename(columns={0: "outside SDPT", 1: "inside SDPT"}).to_dict("index"))
print("   MapBiomas 10 m map's call at those plantations:", pd.crosstab(PL2.in_sdpt.map({1: "inside SDPT", 0: "outside SDPT"}), PL2.group).to_dict("index"))

# ---------------- 6. QC clearings: species at their locations (no labels here; totals only)
q = sp[sp.dataset == "qc"]
print("\n6. Cropin QC clearings: SDPT species at the clearing's point (labels not available here):",
      q.sdpt_species.value_counts().to_dict(), "| within 30 m:", q.sdpt_within_30m.value_counts().to_dict())

# ---------------- outputs
cols = ["pid", "lon", "lat", "state", "CD_MUN", "label", "group", "sdpt_species", "sdpt_within_30m", "sdpt_planted_year",
        "hall_rf_score", "hall_rf_pred", "h4_rf_pred", "full_rf_pred", "n_cuts", "n_cycles", "abrupt_start", "forest_since_1985",
        "last_cut_year", "s_std"]
errs = V[(V.label != V.hall_rf_pred)][cols].copy()
errs.insert(6, "error", np.where(errs.label == 1, "plantation missed", "natural forest called plantation"))
errs.sort_values(["error", "sdpt_species", "hall_rf_score"]).to_csv(RESULTS / "misclassified_points.csv", index=False)
V[cols].to_csv(RESULTS / "val_points_species.csv", index=False)
print("\nwrote", len(errs), "misclassified points")
