# Pine vs eucalyptus: plan (draft, 20 Nov 2026)

Clement Docteur (with Claude). Cropin's new ask: within plantations, tell **pine** (mostly *Pinus taeda* /
*P. elliottii*) from **eucalyptus**. This plan reuses the current pipeline as much as possible. Everything
new goes in `pine_detection/`, and no existing file is edited.

---

## Status (21 Nov 2026)

| Step | State | Where |
|---|---|---|
| SDPT checked as a label source | Done. v2.0 has species for the south: 2013–14 data, about 61% of plantation points covered | `docs/sdpt_check.md`, `src/sdpt_overlay.py` |
| P02: features vs SDPT species | Done. **Nothing separates the species on SDPT labels** (best is latitude, AUC 0.64). The labels are suspect | `docs/p02_findings.md`, `notebooks/P02_species_univariate.ipynb` |
| Gold sample drawn | Done. 250 MapBiomas points: batch 1 (135) audits SDPT, batch 2 (115) adds new labels | `src/make_gold_sample.py` → `data/processed/gold_sample/` (local) |
| Labelling protocol + GEE helper | Done, ready to use | `docs/labelling_protocol.md`, `gee/02_labelling_helper.js` |
| **Photo-interpretation, batch 1** | **To do: two interpreters, about 3–4 h each** | — |
| P01 (agreement, SDPT accuracy), then P03 model | Waiting for labels | — |
| Questions to Cropin (§1) | Not sent yet | — |

---

## 0. Bottom line

1. **Labels are the real problem, not features.** No dataset we have says which species a plantation is:
   - the Cropin QC GeoJSON only has `qc_label` (forest / plantation);
   - MapBiomas has a single class 9 ("forest plantation"), and the 85k validation `.dbf` has no species
     field either.
2. **The cheapest path:** add a species label to plantation points **we already have features for**:
   - about 810 MapBiomas plantation points in `Master Data/master_features_v3*.csv` (all ~40 features and
     AlphaEarth are already there);
   - the 310 QC plantations.

   With that, almost no new GEE export is needed, and every notebook-04, combined-model and harvest-cycle
   feature can be scored again against the species label.
3. **The model is a second stage.** It answers "given that this is a plantation, is it pine or
   eucalyptus?". The two layers then combine:
   `P(pine) = P(plantation) × P(pine | plantation)`.
   The plantation-vs-forest work stays as it is.
4. **Expected strongest signals:**
   - **rotation length and thinning** (eucalyptus is clear-cut every ~6–8 years, pine every ~12–25 years
     with partial thinnings), which is my harvest-cycle/time-series area;
   - **conifer vs broadleaf spectra** (NIR, red-edge, SWIR; H20 already found B11 separates pine from
     broadleaf).

---

## 1. Questions for Cropin first (send this week)

Their answers change the scope, so ask before building much.

| # | Question | Why it matters |
|---|---|---|
| Q1 | **Why** do they need species? Is it for EUDR reporting, harvest-alert timing, or client products? | It decides the metric: accuracy at the pixel, at the alert, or area by municipality |
| Q2 | Do they or their clients (pulp and timber companies) have **species-labelled stands** in PR/SC/RS? | This is the best possible label source and would save days of photo-interpretation |
| Q3 | Two classes, or **pine / eucalyptus / other plantation** (black wattle, planted araucaria, erva-mate)? | Black wattle is common in RS. A forced two-class model will mislabel it |
| Q4 | Output: a 10 m map, one label per alert polygon, or both? Is "unsure" an acceptable answer? | Abstaining on hard cases makes the reported accuracy honest |
| Q5 | Is the species needed **at the alert** (after clearing) or **before** it (standing stand, ≤ 2020 or the year before)? | After clearing, only the history can tell species. This favours the time-series features |

**Default if no answer by 11/24:** three classes (pine / eucalyptus / other-or-unsure), one label per
point or polygon, using imagery from before the clearing.

---

## 2. Labels (the critical path)

### 2.1 Candidate sources, best first

| Source | What it gives | Use | Caveats |
|---|---|---|---|
| Cropin / client stand data (Q2) | Species per stand | Gold labels | May not exist or be shareable |
| **WRI Spatial Database of Planted Trees (SDPT) v2.1** | Planted-forest polygons with a species attribute. Brazil was updated in v2.1 (2026) | Overlay on our existing points to get **silver labels** | Coverage and quality in the south are unknown. WRI says there's no uniform accuracy assessment and the data aren't for very local use. **Check whether it is on GEE** (otherwise download from WRI and upload as an asset) |
| IPEF national planted-forest map (Sentinel-2, 2019–21, eucalyptus / pine / other) | Polygons mapped by photo-interpretation, with omission error 9% for pine and 3% for eucalyptus | Second silver source if the polygons can be obtained | Publication only, so the data may not be public |
| **Our own photo-interpretation** | Gold labels on a stratified sample | Validation set and checking the silver labels | Takes time, so needs a protocol and two interpreters |
| IBGE PEVS (municipal silviculture area by species) | Hectares of eucalyptus and pine per municipality each year | **Check at municipality level only**, never as point labels | Aggregate data |
| GBIF / iNaturalist occurrences | Points of *Pinus* / *Eucalyptus* | A sanity check at most | Noisy, and mostly not plantations |

### 2.2 Recommended protocol

1. **Overlay SDPT** (and IPEF if available) on the MapBiomas plantation points and the QC plantations
   (`pine_detection/gee/01_sdpt_overlay.js`). Record the species, the source and the distance to the
   polygon edge.
2. **Photo-interpret a stratified sample** of about 300–400 points: by state, by SDPT species, and with
   SDPT missing. Use high-resolution imagery (Google Earth Pro historical imagery and Street View where
   there is a road). Cues:
   - pine: dark green, rough star-shaped crowns, often thinned rows, long rotations;
   - eucalyptus: lighter grey-green, smooth uniform canopy, white trunks on Street View, short rotations.

   Two of us label each point blind, and **unsure is allowed**. Report the agreement (Cohen's κ) and the
   SDPT-vs-ours confusion matrix.
3. **Build the label table:**
   - `species_gold`: our photo-interpretation, where both interpreters agree;
   - `species_silver`: SDPT/IPEF;
   - `label_source`.

   Train on silver if it agrees with gold at ≥ 85%, and **always test on gold**.
4. **Keep it out of git.** Species labels on QC polygons are derived from sponsor data, so they go in
   `data/processed/` (gitignored) or the shared Drive, like the master table.

A helper for step 2: a small GEE labelling script (`pine_detection/gee/02_labelling_helper.js`) built from
`gee/cropin/cropin_multipoint_SAG_timeseries_explorer.js`. It would show the point on the satellite basemap
next to its NBR/SAG history, so the rotation pattern helps the interpreter. This is a copy, not an edit.

---

## 3. Hypotheses and features (mostly already computed)

| ID | Hypothesis | Feature (existing, unless marked *new*) | Where it already lives |
|---|---|---|---|
| S1 | **Eucalyptus has shorter rotations** (clear-cut every ~6–8 years vs ~12–25 years for pine) | `rh_n_cycles`, `rh_s_std`, `nc_D_max_nbr_drop`, `nc_D_cc_n_breaks`, `nc_D_lt_n_vertices`; *new*: median years between NBR drops, time since last drop | `Master Data/master_features_v3*.csv`, `public_data/harvest_cycles/series_nbr_*.csv.gz` (1985–2020, 1,831 MapBiomas + 433 QC points), `harvest_cycles/harvest_cycles_findings.ipynb` |
| S2 | **Pine is thinned** (partial NBR drops of 20–40% without a clear-cut) | *new*: count of moderate NBR drops between clear-cuts | Same yearly NBR series |
| S3 | **Conifer vs broadleaf spectra:** needles give lower NIR and red-edge, and a different SWIR | All 10 bands × 12 months | `ref_s2_bands_by_month_*.csv` / `qc_s2_bands_by_month.csv` (gee/05), notebook-04 grid code |
| S4 | **Seasonality:** both are evergreen, but they respond differently to winter frost and dry spells | `nc_B_*_phase_*`, `mm_*_win`, `nc_wi3_NDMI`, `nc_su3_NDMI` | Master table |
| S5 | **Growth rate and structure:** eucalyptus is taller for its age, and pine canopies are rougher | `nc_E_eth_h`, `nc_E_eth_sd50`, `nc_chm_*`, radar `nc_F_VH*`; *new*: height / years since last clear-cut | Master table, Nathan's texture work |
| S6 | **Geography:** pine dominates the cold highlands (Serra Catarinense, Campos Gerais) and eucalyptus the lowlands | *new*: elevation, winter minimum temperature (or frost days), state | One small GEE export |
| S7 | **AlphaEarth embeddings** already separate the two | `ae_A00…A63` | `master_alphaearth_v3.csv`, `cropin_embeddings_2022.csv` |

**Watch-outs:**
- Never use SDPT/IPEF as a **feature** when they are the label source. That would be circular, like the
  MapBiomas-derived features on MapBiomas points.
- S6 can make a model look good by memorizing regions. Always report a **geography-only baseline**, and use
  spatially blocked CV so blocks of the same region aren't split between training and testing.

---

## 4. Analysis (new notebooks, same conventions)

All code imports the shared helpers instead of re-implementing them:
`sys.path.insert(0, str(ROOT / "src")); from qc_analysis import auc_ci, mwu_test, holm, spatial_groups, cv_evaluate, set_style, savefig`.
Figures are saved as `figures/P0x_*.png` through `savefig`. Outputs stay aggregate.

| Notebook | Content | Reuses |
|---|---|---|
| `notebooks/P01_species_labels.ipynb` | Label table, interpreter agreement (κ), SDPT vs gold confusion, class balance by state and by train/QC set | `src/build_mapbiomas_ref_south.py` subset, master `uid`s |
| `notebooks/P02_species_univariate.ipynb` | For each existing feature: AUC [bootstrap 95% CI], Mann–Whitney, Holm across the family. Band × month AUC grid with pine vs eucalyptus as the target. Rotation-length distributions | `auc_ci`, `mwu_test`, `holm`, notebook-04 grid code (copied into `pine_detection/src/`) |
| `notebooks/P03_species_model.ipynb` | Compare (a) geography only, (b) sparse logistic regression on 5–10 features (interpretable, Cropin's preference), (c) random forest + AlphaEarth. Spatially blocked CV (20 KMeans blocks, seeds 0–19). Metrics: AUC, balanced accuracy, recall per species, and the share left "unsure" for a chosen confidence band | `cv_evaluate(..., kind="group", groups=spatial_groups(d))`, `make_model`, combined-model `search_v2.py` structure |
| `notebooks/P04_species_transfer_qc.ipynb` | Train on MapBiomas points, test on QC plantations (gold-labelled subset). **Label-free check:** predicted pine share by municipality vs IBGE PEVS share (Spearman ρ). Combined layer `P(pine)` | Same as the plantation work's "reference → QC" transfer test |
| `notebooks/P05_species_and_false_alarms.ipynb` *(stretch)* | Does knowing the species help the **main** goal? The error clusters of Phase 6 are in araucaria/pine country. Test a pine-specific plantation-vs-forest model against araucaria forest | `cv_evaluate` nested ≤ 5%-forest threshold, unchanged |

**Which error matters:** for `cv_evaluate`'s nested threshold, label `1 = eucalyptus, 0 = pine` (or the
reverse, depending on Q1), and set `max_share` on the class Cropin least wants mislabelled. Until we know,
report AUC and balanced accuracy, and leave the thresholded numbers as secondary.

---

## 5. GEE (new scripts only, in `pine_detection/gee/`)

| Script | Purpose | Pattern copied from |
|---|---|---|
| `01_sdpt_overlay.js` | Species (and distance to edge) from SDPT/IPEF at `mapbiomas_ref_south` points and `cropin_qc` polygons → `species_overlay_{ref,qc}.csv` | `gee/03` (reads the same assets, exports to `Drive/cropin_exports`) |
| `02_labelling_helper.js` | Satellite basemap + NBR/SAG history per point for photo-interpretation | `gee/cropin/cropin_multipoint_SAG_timeseries_explorer.js` |
| `03_terrain_climate_export.js` | Elevation (SRTM/Copernicus DEM), ERA5-Land winter minimum temperature (or frost days) at the same points (S6) | `gee/05` (point buffer, split by state) |
| `04_extra_points_bands.js` *(only if we add new labelled points)* | Same monthly band medians as gee/05, for a new `species_points` asset | `gee/05`, with the asset name as a parameter |

The generated scripts that embed coordinates stay out of git, like `combined_model/` does.

---

## 6. Timeline (today is Fri 20 Nov; to Cropin on 12/1, final on 12/10–11)

| Dates | Step | Output |
|---|---|---|
| 20–24 Nov | Send Q1–Q5. Check SDPT (GEE or download) and the IPEF data. Write the photo-interpretation guide | `pine_detection/docs/labelling_protocol.md` |
| 23–26 Nov | SDPT overlay. Two interpreters on ~300 points. P01 | Label table, κ, SDPT accuracy |
| 25–28 Nov | P02 univariate + rotation-length feature (S1/S2) | Feature ranking, figures |
| 27–30 Nov | P03 first model (sparse logistic regression + random forest) | **Preliminary result for 12/1**, clearly marked exploratory |
| 1 Dec | Deliver to Cropin: feasibility, label quality, first accuracy, open questions | One page in `pine_detection/docs/` |
| 2–8 Dec | Act on Cropin's feedback. P04 QC transfer + IBGE municipal check. Optional P05 | Final numbers |
| 9–11 Dec | Final write-up. Add a pine/eucalyptus section to the team summary (with the teammates' OK, since it's a shared file) | Final deliverable |

**Fallback if labels come in late:** deliver P02 (which features separate the species, on SDPT silver
labels), with an honest note on label quality. Don't present a map.

---

## 7. Risks

- **Label noise:** SDPT may be old or wrong in the south. Mitigation: gold test set, and report SDPT's own
  accuracy.
- **Class imbalance and confounding:** pine is concentrated in the SC/PR highlands, so species and region
  are mixed up. Mitigation: geography-only baseline, spatial CV, results by state.
- **"Other" plantations** (black wattle, araucaria, erva-mate) get forced into two classes. Mitigation: a
  third class or abstaining (Q3).
- **Post-clearing alerts have no canopy left.** Use pre-clearing composites (2019–22, as gee/05 does) and
  the harvest history.
- **Small QC gold set:** wide confidence intervals. Report bootstrap CIs and don't oversell.
- **Privacy:** no per-polygon species labels or coordinates in the public repo.

---

## 8. Proposed folder layout

```
pine_detection/
  PLAN.md                     this file
  README.md                   short map of the folder (once there is content)
  docs/labelling_protocol.md  photo-interpretation guide
  docs/findings.md            results for Cropin
  gee/01…04_*.js              new GEE scripts (copies of patterns, no edits to gee/)
  src/species_features.py     rotation-length and thinning features, band x month grid (copied), species palette
  notebooks/P01…P05_*.ipynb
```

**Species palette** (kept apart from `qc_analysis.PAL`, so plantation/forest colours keep their meaning):
pine `#2F5D50` (dark teal-green), eucalyptus `#C08A2B` (ochre), unsure/other greys. Check colour-blind
separation with the same ΔE check before use. No bright red.

## Sources
- WRI, [Spatial Database of Planted Trees v2.0](https://www.wri.org/research/spatial-database-planted-trees-sdpt-version-2) and [v2.1 dataset page](https://datasets.wri.org/datasets/spatial-database-of-planted-trees-sdpt-version-21) (Brazil updated)
- GFW blog, [Updated planted trees map](https://www.globalforestwatch.org/blog/data-and-tools/updated-planted-trees-map-near-global-coverage/)
- IPEF national planted-forest mapping, [Série Técnica 48](https://www.ipef.br/publicacoes/stecnica/nr48/ipef-sertec-48-079.pdf)
- UDESC master's thesis, Pinus/Eucalyptus/clearcut mapping in SC with Sentinel-2 + random forest ([summary](https://www.udesc.br/arquivos/cav/id_cpmenu/2415/Resumo_disserta__o_mestrado_16044142407719_2415.pdf))
