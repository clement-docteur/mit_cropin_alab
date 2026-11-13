# Harvest cycles: project handoff

Everything someone needs to pick up the harvest-cycle work: the goal, the data rules, what was done, the results,
the decisions taken and why, what is in progress and what comes next. Written from a long working session between
the project lead and an AI coding assistant (Claude Code), so that others can continue without that conversation.

*Last updated: 13 Nov 2026.* Update the "Status" section when you change something.

---

## 1. Goal

- **Project:** MIT Analytics Lab × Cropin. Telling tree **plantations** apart from **natural forest** in Paraná,
  Santa Catarina and Rio Grande do Sul (southern Brazil).
- **Why:** Cropin flags deforestation for the EU Deforestation Regulation (EUDR: no goods from land deforested after
  **31 Dec 2020**). Plantation harvests look like deforestation and raise false alarms. Hiding real deforestation
  (calling natural forest a plantation) is the costly error.
- **This workstream:** the *harvest-cycle* hypothesis. Plantations are clear-cut and replanted in rotations; natural
  forest is not. Does each pixel's own Landsat history (1985–2020) show that? Only imagery up to the EUDR cutoff is used.

## 2. Data and confidentiality rules (read first)

| Data | What | Where | Rule |
|---|---|---|---|
| MapBiomas validation points | 1,831 points of the team's table (`data_exploration/cropin_model_features_2022.csv`), 429 plantation / 1,402 natural forest (2022 reference class); public | `public_data/harvest_cycles/points.csv` (pid `V…`) | Labels are public |
| Cropin QC polygons | 433 hand-checked cases: 310 plantation harvests flagged as deforestation (false alarms), 123 real forest clearings | GeoJSON in `CROPIN-MIT-DATA/cropin_qc_data/` (git-ignored) | **Labels are confidential: never commit them, never output per-polygon values.** Locations are *not* confidential (confirmed by the project lead). |
| Derived public data | Yearly Landsat series, Hansen values, MapBiomas class history, Meta canopy height at all 2,264 points; QC label column left empty | `public_data/harvest_cycles/` | Safe to commit |

- The repository is **public**. `CROPIN-MIT-DATA/` is git-ignored; raw downloads and anything with QC labels stay there.
- QC point ids `Q000…Q432` follow the order of the polygons in the GeoJSON; the notebook checks that each point lies in
  its polygon's bounding box before reading the labels.
- Midterm baselines on the QC set (from the team's slides, not recomputed): MapBiomas (Collection 9) catches 4% of
  plantations while flagging 2% of forest; Google Natural Forests (cut-off 0.15) catches 23% while flagging 6%.

## 3. Method in one page

- **Imagery:** Landsat 4/5/7/8 **Collection 1**, top-of-atmosphere reflectance, read pixel by pixel from Google's public
  bucket (`gcp-public-data-landsat`, no Earth Engine). About 6 low-cloud scenes per path/row and year (3 dry season
  Apr–Sep, 3 wet Oct–Mar); 8,063 scenes; median 6 clear observations per point-year. Cloud mask from the QA band.
  Landsat 4/5 and 8 shifted onto the Landsat 7 scale (offsets from same-point pairs ≤ 8 days apart).
- **Index:** NBR = (NIR − SWIR2)/(NIR + SWIR2); one value per point and year = median of clear observations.
- **Levels (from Hansen tree cover, never from labels):** closed canopy **0.584** (5th percentile of 2000–2020 NBR at
  stable ≥ 90% tree-cover pixels); open land **0.405** (25th percentile of 1999–2001 NBR at ≤ 10% tree cover).
- **Segmentation:** despike one-year spikes, then LandTrendr-like piecewise-linear fit, up to **18** segments, chosen by
  BIC (LandTrendr's usual 6 fits only ~2 cuts).
- **Events:** *cut* = fall ≥ **0.20** NBR within ≤ **3** years between segment ends; *cycle* = cut that regrows **50%**
  of the fall within **10** years; *regrowth time* = years to regrow **80%**; *abrupt start* = ≥ 3 open-land years then
  closed canopy within 5 years; *forest since 1985* = no cut and never below closed canopy.
- **Feature sets:** H4 = n_cycles, regrow_speed, forest_since_1985, abrupt_start. H-all = H4 + n_cuts,
  years_since_last_cut, max_drop, s_frac_below_forest, s_min, s_max_drop3. Variability = s_std (picked after first
  results: treat as needing confirmation). Yes/no: has_cut, has_cycle, two_plus_cycles, fast_regrowth.
- **Evaluation:** train on MapBiomas points only; spatial cross-validation (GroupKFold by municipality, 5 folds × 5
  repeats); operating threshold = flag ≤ 5% of natural forest; Cropin QC used **only as an independent test**.
  P-values clustered by municipality (MapBiomas) or 0.1° cell (QC); AUC intervals by resampling those clusters.

## 4. Main results

All in `harvest_cycles/harvest_cycles_findings.ipynb` unless marked *(session analysis)*.

| | MapBiomas points | Cropin QC |
|---|---|---|
| At least one cut, plantation vs forest | 76% vs 14% | 35% vs 20% |
| At least one cycle (clustered p) | 72% vs 14% (p ≈ 3×10⁻⁹⁸) | 34% vs 19% (p = 0.006) |
| Two or more cycles | 39% vs 2.2% | 10% vs 7.3% (not significant) |
| Abrupt start | 45% vs 1.1% | 14% vs 4.1% |
| Forest since 1985: forest protected / plantations wrongly vetoed | 45% / 1.6% | 30% / 6.1% |
| Regrowth time (median, years) | 1.7 vs 1.8 (no difference) | 1.4 vs 2.2 (no difference) |
| Variability alone, AUC | 0.97 | 0.76 |
| Number of cuts alone, AUC | 0.84 | 0.58 |

| Model (AUC; MapBiomas LR / RF spatial CV, QC = trained on MapBiomas) | MapBiomas | QC |
|---|---|---|
| H4 | 0.908 / 0.904 | 0.675 / 0.673 |
| H-all | 0.947 / 0.945 | 0.757 / 0.743 |
| H-all + variability | 0.973 / 0.969 | 0.796 / 0.775 |
| 36 yearly values + changes (RF) | 0.973 | 0.812 (catches 43% of QC plantations at 5.7% QC forest flagged) |
| Hansen alone | 0.933 / 0.916 | 0.697 / 0.684 |
| H-all + Hansen | 0.964 / 0.959 | 0.773 / 0.766 |
| Team's features (2022 imagery) | 0.909 / 0.919 | – |
| Team's features + H-all | 0.969 / 0.969 (all 25 folds better) | – |

- **Plantations caught at ≤ 5% forest flagged (MapBiomas, RF):** H-all 78%, 36 yearly 90%, team 64%, team + H-all 88%
  vs the MapBiomas 10 m map 86% (no significant difference); team + H-all catches 26 of the 39 plantations the map calls forest.
- **Why QC is harder:** median QC polygon **269 m²** (under a third of a 30 m Landsat pixel); plantation polygons show a
  cut 33–36% of the time up to 2,700 m², 64% above (n = 11); only 8% of QC plantations show a cut in 2010–2020;
  QC forest is more disturbed (70% fail "forest since 1985" vs 55% on MapBiomas).
- **Veto "forest since 1985":** on the MapBiomas 10 m map it cancels 29 of 72 wrong plantation calls and loses 5 of 369
  correct ones; adds nothing inside a harvest-history model. Use only to cancel a plantation call.
- **Best use of the series:** continuous summaries beat counts (variability vs cut count +0.13 AUC MapBiomas, +0.19 QC);
  NBR beats NDVI (+0.07); 6 segments too few (−0.06); dry-season composite gives more false cuts in forest (21% vs 14%);
  yearly maximum misses cuts (−0.07); drop threshold best near 0.15–0.20.
- **Checks:** where Hansen records a loss, the segmentation finds a cut 85% of the time; abrupt start catches 52% of
  plantations planted on open land; canopy closure falls a median of 0 years from the reference planting year.
- **Section 8, against Cropin's own tools (stand-ins):** inside an approximate Cropin forest layer (MapBiomas 10 m forest
  + Meta canopy ≥ 4 m; 1,124 forest points) the harvest history flags ~3% of natural forest (QC forest: 5.7–6.5%).
  Team's SAG alone 0.870 / 0.862 AUC and catches 45% of plantations; **SAG + H-all 0.968 / 0.966, catches 87%**,
  all 25 folds better; finds 192 of the 237 plantations SAG misses (SAG values are 2022 imagery: not EUDR-pure).
- *(session analysis)* **Seasonality ("winter drop")**: wet-minus-dry season difference from the dated observations;
  natural forest is more seasonal. NBR 2013–2020 version: AUC 0.83 MapBiomas / 0.70 QC (robust within each state and to
  a latitude trend); the pre-specified NDVI all-years version is weak (0.69 / 0.59). Adding the four seasonal versions
  to H-all: +0.04–0.05 AUC on QC (CI excludes 0), +0.01 on MapBiomas; +0.008 (n.s.) on top of the 36 yearly values.
  Picked after looking: needs confirmation. These four features are now in the feature script (section 7).
- *(session analysis)* **Training on 70% of QC + MapBiomas, testing on the other 30%** (20 grouped splits): no gain
  (≤ +0.01 AUC); training on QC alone does about as well; one split alone is very noisy (0.68–0.85 for one model).
  Decision: keep training on MapBiomas and testing on QC.

### Things that did not work (or not tested)
Regrowth speed; cycle counts on QC (AUC 0.58); two or more cycles on QC; NDVI instead of NBR; dry-season-only or yearly
maximum composites; LandTrendr's default 6 segments; drop thresholds 0.10 (a third of forest shows a cut) or ≥ 0.25;
3×3 neighbourhood instead of the single pixel (QC 0.75 vs 0.80); ignoring 1985–1992 in the veto; first-rotation
plantations established 2014–2020 (cut seen in 44%); adding QC cases to training. Pine vs eucalyptus: **not tested**
(no species labels).

## 5. Decisions and why

- **Landsat, not Sentinel-2:** harvest cycles need decades (rotations 6–20+ years); Sentinel-2 (10 m) only has ~4 years
  before the cutoff. Use Sentinel-2 for recent, small-polygon signals (SAG, texture) and combine.
- **No Fourier series on the yearly data:** harvests are irregular one-off steps, 0–2 per pixel; segment-based methods fit
  them better. Fourier/harmonics only make sense for the within-year season on monthly data (CCDC-style).
- **Monthly Landsat:** full archive is ~3.3× the scenes; only from ~1999 are most months covered. Worth it only for
  seasonal features, preferably in Earth Engine for 2013–2020.
- **QC data biases** (raised by the project lead; answer: change reporting, not training): QC has no forest outside
  Cropin's layer; QC forest is forest being cleared; QC plantations are stands at harvest; small and clustered numbers
  (use spatially blocked intervals, remove same-pixel duplicates); labels may share Cropin's model errors; mostly mixed
  pixels (report by pixel coverage); unknown holdout selection. Open question: is the main use an alert filter (QC is
  the right test) or a 10 m map (judge forest errors on MapBiomas)?
- **Feature pipeline (Nov 2026):** two scripts (preprocessing; features), thresholds in `config.py` with command-line
  overrides recorded next to every output, notebook imports the feature code (single source of truth), MapBiomas label
  in the features CSV as `label_mapbiomas`, raw downloads git-ignored, our 2,264 points first. Unit tests and pinned
  dependencies deliberately skipped for now. Thresholds are *inputs*, not many feature variants; tune later with
  nested cross-validation (tuning on MapBiomas risks fitting its labels).

## 6. Before production (to do)

- **Move the pipeline to Google Earth Engine with Landsat Collection 2.** Collection 1 (used here) is deprecated by USGS;
  Google's bucket still serves it but may stop. Collection 2 needs re-validating every feature and threshold (NBR
  levels, harmonisation offsets). `harvest_cycles/gee/landtrendr_harvest_features.js` (on the analysis branch) already
  computes the features with real LandTrendr on Collection 2.
- Tune thresholds with nested spatial cross-validation; confirm variability and the seasonal features on new data.
- Get Cropin's exact forest layer and SAG score: run `harvest_cycles/gee/cropin_layers_at_points.js` in Earth Engine and
  save its two CSVs in `public_data/harvest_cycles/`; the notebook's section 8 then uses them (rewrite its text).

## 7. Repository map

| Where | What |
|---|---|
| `main` | `harvest_cycles/harvest_cycles_findings.ipynb` + `public_data/harvest_cycles/` (merged via PRs #3 and #4); team folders; `Google Earth Engine Base/` (Cropin's forest-layer and SAG scripts) |
| branch `claude/harvest-findings-notebook` | Latest notebook: section 8 (Cropin forest layer / SAG), `harvest_cycles/gee/cropin_layers_at_points.js`, `public_data/harvest_cycles/canopy_height.csv`, this file, and the feature pipeline below. **Not merged yet.** |
| branch `claude/friendly-cori-fvrwiz` | Original full pipeline: `harvest_cycles/scripts/` (index → points → scenes → extraction → Hansen → series → features → analysis), `results/`, `figures/`, `docs/summary.md`, `gee/landtrendr_harvest_features.js`. Superseded by the new scripts once they are verified; not merged. |
| `harvest_cycles/config.py` *(being added)* | All thresholds and settings (defaults reproduce the analysis) |
| `harvest_cycles/build_harvest_cycles_features.py` *(being added)* | Yearly series + dated observations → features CSV (19 features + `label_mapbiomas`); also imported by the notebook |
| `harvest_cycles/preprocess_landsat.py` *(being added)* | Points CSV (pid, lon, lat) → Landsat scenes → clean dated observations → yearly series (resumable; raw cache git-ignored) |

Outside the repository (owned by the project lead): an HTML explainer page and a 13-slide deck (private claude.ai
artifacts), and .pptx slides for the 12 Nov findings meeting.

## 8. Status (13 Nov 2026)

- Done and pushed: notebook sections 1–10 (reproduces the original pipeline exactly: 424 automated checks against the
  committed results); section 8 (stand-ins for Cropin's layers); Earth Engine export for the exact layers; canopy height.
- **In progress on `claude/harvest-findings-notebook`:**
  - `config.py`, `build_harvest_cycles_features.py`: written; the features match the notebook's exactly (max difference
    0.0 on all 15 series features; the 4 seasonal features identical to the session analysis).
  - `preprocess_landsat.py`: being ported from the analysis branch's `scripts/` (steps 1–6 + optional Hansen sampling).
  - Notebook refactor to import the feature module; README with run instructions and the production note.
  - Planned verification: re-run the notebook (424 checks), run preprocessing end to end on the 2,264 points and compare
    with the existing series, independent review.
- Next ideas: combine harvest history with the team's height / Natural Forests / land-history features; QC reporting
  with blocked intervals and by pixel coverage; check QC labels against independent maps; ask Cropin the open questions
  (main use, holdout selection, an "unsure" label, SAG/CCDC settings).

## 9. Working conventions

- Commit to the branches above only; open pull requests only when asked.
- Never commit Cropin labels or per-polygon QC results; aggregates only.
- Keep notebook text and slides short and plain; every number in prose must match an output.
- Discuss scope and questions before writing code when asked to; report what was verified and how.
