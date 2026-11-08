# Feature Exploration – 00 Workflow Plan

*Prepared by Nathan Chu (with Claude) · Version 4.1, 5 November 2026 · Claude does the implementation; Nathan runs GEE batch scripts and approves at each checkpoint · **[inference]** = our interpretation, not tested · Hypothesis numbers refer to* Task 1.2 – Hypotheses Master List

---

## 1. Goal

Test every Sentinel-2-derived and supporting feature family against the hypotheses in *Task 1.2 – Hypotheses Master List* (H1–H43, M1–M6, HC1–HC10) and against Cropin's methods. Find the smallest set of interpretable features that raises plantation recall without calling real forest "plantation".

**Not in scope:** black-box models, tree or clearing detection, paid imagery, and production integration.

## 2. Terms

| Term | Meaning here |
|---|---|
| Plantation recall | Share of true plantation points we label plantation |
| Forest error | Share of true forest points we label plantation. This is our dangerous error. It is not Cropin's "false positive", which is a plantation harvest flagged as deforestation. |
| Operating point | Recall measured at a fixed forest error (1%, 2%, or 5%) |
| Family | A group of related features (A–H below) |
| Lockbox | Data that stays untouched until the final model is frozen, then is scored once |

## 3. Decisions (fixed)

- Labels follow the EUDR ([Regulation (EU) 2023/1115, Art. 2](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX%3A02023R1115-20251226)):
  - **Forest:** wild or self-seeded pine, restoration plantings, shade-grown erva-mate, bracatinga, and abandoned plantations.
  - **Plantation:** planted araucaria and black wattle.
  - Source of these team decisions: *Task 1.2 – Hypotheses Master List*, §1 and sections F and Hard cases.
- No black-box models. Logistic regression and shallow trees only. AlphaEarth appears only as a comparison benchmark.
- Clean slate: we don't build on Cropin's layers, but we compare against them.
- Nathan works independently of teammates, and we cross-check results afterwards.
- The cutoff year is a parameter (EUDR = 2020).

## 4. Roles and execution

| Who | Does |
|---|---|
| Claude | Writes all GEE scripts, one per batch, each queuing all its exports. Reads results from the synced `cropin_texture` folder, then does all statistics, models, the run log, and docs. Checks that every expected file arrived and is complete. Labels points only before looking at their feature values. |
| Nathan | Runs each batch script in the GEE Code Editor and starts its tasks in the Tasks tab (about 2–5 minutes per batch). Second labeller for visual checks and hard cases. Approves decisions at each wave checkpoint. |
| Independent reviewer agent | At the end of each wave, gets the code and data but not Claude's conclusions, and re-computes the key numbers. |

**Execution path.** Exports go to Google Drive (`cropin_texture`). Google Drive for desktop syncs that folder to Nathan's Mac, where Claude reads it. Each script's header lists the files it will produce, roughly how long the exports take, and exactly what to click. If GEE Python API access becomes possible later, Claude takes over running the scripts, and nothing else in the plan changes.

## 5. Data and labels

| Set | Definition | Use |
|---|---|---|
| MapBiomas points (7,535 in PR/SC/RS) | Forest = class 3 every year 2013–2022. Plantation = class 9 every year 2013–2022, so both classes use the same window. Points within 100 m of each other are deduplicated. Agreement count and border flag are kept. | Training and blocked CV |
| MapBiomas lockbox | ~20% of 0.5° blocks, stratified by state, chosen at random in Phase 0 | Scored once, at the end |
| Alert-like subset | MapBiomas points that look like Cropin alerts (inside a forest mask, small patches). Built from Wave 1 onward (decision 5 Nov 2026); method in `Run Log.md`. | Checks that results transfer to alert conditions, and calibrates thresholds |
| Cropin QC (433 polygons: 310 plantation, 123 forest) | Split into dev and held-back halves by spatial block. About 100 labels visually checked by two labellers. | **Evaluation only, never training.** The dev half is used only to look at errors, never to tune. |
| Hard-case set | HC1–HC10 examples gathered by hand | Qualitative results only (n and examples) |

Weights: report results both unweighted and weighted by PESO_AMOS.

Low-agreement points (decision 5 Nov 2026): MapBiomas points where only 1 interpreter agreed with the 2022 class (`agree` = 1; 55 dev points: 22 forest, 33 plantation; 47% on borders) are **dropped from training but kept in evaluation**, where they are reported as their own group next to edges. The same rule applies to the lockbox. Phase 5 checks how sensitive results are to this choice.

Notes:

- The MapBiomas validation file has its coordinate fields swapped: `LON` holds latitude and `LAT` holds longitude (our check, 4 Nov 2026).
- The ~60 forest points in the QC held-back half are too few to confirm a forest error of 5% or less. If we observe 5% on 60 points, the 95% CI runs from about 1% to 14% (our calculation). QC results show direction, not proof. We will ask Cropin to score our layer on their blind holdout.

## 6. Validity rules

1. No leakage. Family D uses only satellite signals, never MapBiomas class history.
2. Both classes use the same stability window (2013–2022).
3. Deduplicate points within ~100 m. Spatially blocked CV (0.5°, GroupKFold).
4. About 100 QC labels are visually checked, using the two-labeller protocol.
5. Labels follow the legal definitions in §3.
6. Results are checked on the alert-like subset and on QC dev, never on the lockbox before the end.
7. As-of rule: features come only from before clearing. If the date is unknown, the input is tagged "date unknown" (e.g., Meta canopy height, AlphaEarth). On QC points, results are also reported without these inputs.
8. Until Cropin sends QC clearing dates, use the season before the earliest possible clearing. Drop points where NDVI shows the stand was already gone, and record how many were dropped.
9. Report n and 95% CIs everywhere. Recall and forest error are always reported together.
10. Data checks before any analysis: coordinates, class codes, row counts, and null counts.
11. Versions logged for scripts, collections, and date windows.
12. Every external layer is tagged "MapBiomas-derived: yes / no / unknown". Derived or unknown layers are judged mainly on QC points, and the tag is shown in every results table. (Whether Natural Forests of the World used MapBiomas labels is **unverified**.)
13. Store the clear-observation count per point and stratify results by it. Clouds stay in the data and are handled through features and stratification.
14. Texture is reported separately for every window size. QC polygons are tiny (~0.02 ha, about 2 pixels), while windows reach 70 m, so large windows measure the neighbourhood, not the stand **[inference]**.

## 7. Feature families and hypothesis coverage

| Family | Features | Hypotheses | Wave |
|---|---|---|---|
| A. Spectral | Seasonal-composite bands and indices, SAG, SWIR | H19, H20, H24 (spectral part), SAG | 1 |
| C. Texture / uniformity | GLCM, local standard deviation, SNIC segment stats, directional texture/FFT | H18, H24, H15, H16, H33 | 1 |
| D. History | Landsat 1985+ and Sentinel-2 time series, CCDC, LandTrendr | H1–H9, H11, HC3, HC8 | 1 |
| B. Seasonal | Seasonal amplitude, timing, drought response, thermal/ET | H21, H22, H23, H27 | 2 |
| F. Radar | Sentinel-1 C-band, ALOS PALSAR L-band | H25, H26 | 2 |
| E. Structure | Meta CHM, GEDI, height spread, height vs. age | H12, H13, H14, H17 | 3 |
| G. Terrain / context / geometry | Slope, riparian position, stand shape and size, neighbourhood, distances | H29–H32, H34–H38, H43 | 3 |
| H. External / registry | NFW, JRC GFT, Forest Typology, plantation-year products, CAR, registries, FSC, municipal stats | H10, H39–H42 | 3 |
| (Benchmark only) | AlphaEarth | H28 | Phase 7 |
| (Error analysis only) | — | M1–M6 | Phase 6 |

Waves only set the order. **Every family is tested.** Wave 1 holds the families with the strongest evidence so far (SAG, SWIR, and texture results from 4 Nov 2026; see *2026-11-04 Cropin Meeting Brief*).

H16 (planting rows): typical row spacing is about 3 m, and a 10 m pixel can only resolve patterns of about 20 m or more, so rows are probably invisible to Sentinel-2 **[inference, sampling theory]**. We run one cheap test (directional texture and FFT) and stop unless it shows a signal.

## 8. Phases

| Phase | Work | Checkpoint output |
|---|---|---|
| 0. Setup | Data checks, recount under §5, dedup, set the lockbox, QC split, re-run the texture and SAG baselines under the new rules (earlier results used unequal stability windows and no dedup) | Clean datasets, new baseline numbers, run log started |
| 1. Wave 1 | Families A, C, and D: single-feature screens, then a within-family model | Family reports and reviewer check |
| 2. Wave 2 | Families B and F | Same |
| 3. Wave 3 | Families E, G, and H | Same |
| 4. Combine | Stability selection across the families that passed. Cap of ~10 features; trees of depth 3–4 at most. | Candidate model(s) |
| 5. Robustness | By year and season, leave-one-state-out and leave-one-biome-out, by clear-observation count, by window size | Robustness table |
| 6. Error breakdown | Border vs. interior, biome, agreement count, state, spatial clusters, hard cases, M1–M6 | Error report (also feeds Task 1.5) |
| 7. Final | Freeze the model, then score once on the lockbox and QC held-back. Compare against baselines and AlphaEarth. | Final results and memo |

At each checkpoint Nathan reviews the results and decides whether to change the plan before the next phase.

## 9. Decision rule (per feature)

A feature passes if **all** of these hold:

1. Mann–Whitney test, Benjamini–Hochberg-corrected within its family: $q < 0.05$.
2. On blocked CV, the lower end of the AUC's 95% CI is above $0.60$.
3. The direction is the same across states and matches what the hypothesis predicts. If MapBiomas and QC-dev disagree, the feature is flagged "context-dependent".
4. It adds at least +5 pp recall at 5% forest error over the baseline, and does not get worse at 1% or 2%. On MapBiomas points the baseline is the NFW-free one (§10).

Operating points: Cropin says the tolerance for missed forest is "technically zero" (project file *GMT20261014-123031 transcript*). We therefore report recall at 1%, 2%, and 5% forest error. The 1% point rests on about 15 forest points per evaluation, so it is noisy.

## 10. Baselines

| On MapBiomas points | On QC points |
|---|---|
| **Decision-rule baseline:** SAG + height + height spread (no NFW, no land history) | All four signals + SAG |
| Reported for comparison: SAG + NFW + height + height spread | SAG + NFW + height + height spread |
| MapBiomas map; NFW at several thresholds; SAG alone | Same, plus Cropin's current outcome |

Each baseline carries its "MapBiomas-derived" tag (rule 12).

Decision 5 Nov 2026: NFW is excluded from the MapBiomas decision-rule baseline. In Phase 0, NFW scored AUC 0.957 on MapBiomas dev but 0.726 on QC dev (`Run Log.md`), and its training labels may be MapBiomas-like (unverified). NFW is still reported as a comparison on both sets.

## 11. Outputs and record keeping

Everything goes in `Cropin - A Lab/Feature Exploration/`:

- `00 Workflow Plan` (.md + .docx)
- `Run Log.md`, with one entry per run: date, script version, data window, n, results, decisions, and GEE cost
- `scripts/`
- `results/`
- One report per family (.md + .docx), with inline sources

Results go to the team after each wave.

## 12. Standing instructions

1. Recall and forest error are always reported together, at all three operating points.
2. Edges and mixed pixels are reported as their own category.
3. Every evaluation includes a reminder that Cropin can't see forest it misses.
4. Report GEE cost (EECU-seconds or run time per 1,000 points) next to AUC.
5. Use an inline source for every external claim, and tag inferences.
6. Don't use NICFI or Google Earth imagery as model input. They are for visual checks only.
7. Don't open the lockbox before Phase 7.
8. New facts relevant to the master list get added there, and Nathan is told.
9. Drafts are reviewed in chat before they are saved.
10. Clouds stay in the data, and are handled through features and stratification.
11. Every model report includes the **threshold-transfer check**: the threshold that gives 5% forest error on MapBiomas, applied to QC dev, with the resulting QC forest error and recall (decision 5 Nov 2026; in Phase 0 this gave 15–17% forest error on QC).

## 13. Open items

| Item | Owner |
|---|---|
| GEE Python API access from Claude's environment (blocked as of 5 Nov 2026; batch workflow in §4 used instead) | Nathan |
| QC clearing dates | Deepak (Cropin) |
| Was Natural Forests of the World trained on MapBiomas labels? | Claude (check documentation) |
| AlphaEarth as benchmark only: confirm this matches Project Plan Task 1.8 | Nathan |
| What "O2" meant in the v3 decision rule (replaced in v4 by "matches the hypothesis") | Nathan |
