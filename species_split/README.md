# Pine vs eucalyptus: which plantations does the harvest-cycle model get wrong?

Pine and eucalyptus are cut on different schedules (eucalyptus about every 6–7 years, pine every 15–25 years with
thinnings), so it seemed worth checking whether one of them is over-represented among the points that
`harvest_cycles/harvest_cycles_findings.ipynb` misclassifies. If so, the two might need to be detected in different ways.

**Short answer: no.** Pine and eucalyptus are missed at the same rate (18% vs 16%). Their share among the errors matches
their share in the sample, and their Landsat histories look alike. The split that does matter is between **large
industrial plantations and the rest**. Plantations outside the industrial-plantation map are missed twice as often,
and almost all of Cropin's QC clearings fall outside it.

## Where the species come from

MapBiomas's land-cover maps have a single "forest plantation" class with no species. The species map used here is the
**Spatial Database of Planted Trees (SDPT v2.0)** from WRI / Global Forest Watch. For Brazil it holds Transparent
World's mapping of *large industrial plantations* from 2013–14 imagery (Petersen et al. 2016), with each polygon
labelled pine, eucalyptus, acacia (black wattle), unknown, or young plantation / clearing.

`scripts/extract_sdpt.py` streams SDPT's public 7.7 GB file and keeps only:
- the polygons around our points (`public_data/sdpt/sdpt_polygons_at_points.geojson`);
- the planted area by species in the three states (`public_data/sdpt/sdpt_area_by_species.csv`).

A point gets the species of the SDPT polygon that contains it, or "none".

**Model calls:** `scripts/notebook_predictions.py` reruns the notebook's own code to get each MapBiomas point's
out-of-fold score. It uses the first cross-validation split, at the notebook's threshold of at most 5% of natural forest
flagged. It reproduces the notebook's figures exactly: 78% of plantations caught by H-all (random forest), 66% by H4,
90% by the 36 yearly values. The main model below is **H-all, random forest**.

## Findings (MapBiomas reference points)

All numbers are in `results/species_analysis.txt`.

**1. The sample is not skewed toward one species.** Of the 428 plantation points, 168 fall in SDPT pine, 87 in
eucalyptus, 18 in acacia or young stands, and 155 (36%) outside any SDPT polygon. Inside SDPT, the pine : eucalyptus
ratio of the points (66 : 34) matches the mapped planted area within 30 km of the points (66 : 34, binomial
p = 0.95). The same holds by state: Paraná is mostly pine, Santa Catarina and Rio Grande do Sul are more mixed.

**2. Neither species is over-represented among the errors.**

| Plantations missed | Pine | Eucalyptus | Not in SDPT |
|---|---|---|---|
| H-all, random forest (main model) | 30 / 168 (18%) | 14 / 87 (16%) | 50 / 155 (32%) |
| H4, random forest | 29% | 30% | 46% |
| 36 yearly values, random forest | 4% | 8% | 18% |

- **Pine vs eucalyptus:** odds ratio of being missed 1.13 (95% CI 0.57–2.27); Fisher p = 0.86, p = 0.73 with errors
  clustered by municipality. No model shows a significant difference.
- **Among the 95 missed plantations:** 32% pine, 15% eucalyptus, 53% outside SDPT. Among the 333 caught: 41%, 22% and
  32%.

**3. In the Landsat record, SDPT pine and eucalyptus behave alike.** The table covers MapBiomas plantation points
(NBR sd: the variability of yearly NBR).

| | A cut | 2+ cycles | Mean cuts | Abrupt start | Years between cuts (median) | Years to regrow 80% | NBR sd |
|---|---|---|---|---|---|---|---|
| Pine | 82% | 39% | 1.58 | 49% | 11 | 2.4 | 0.177 |
| Eucalyptus | 75% | 44% | 1.55 | 47% | 11 | 2.5 | 0.170 |
| Not in SDPT | 69% | 32% | 1.26 | 37% | 8 | 1.8 | 0.138 |

- **No difference between the two:** nothing separates pine from eucalyptus (all p > 0.1). About 40% of cut intervals
  are 8 years or less for both.
- **What that could mean:** either the two are managed on similar schedules in the south (eucalyptus grown longer for
  sawlogs and energy, pine thinned in between), or SDPT's 2013–14 species labels are unreliable at single points.
  This analysis cannot tell which.
- **Either way:** a species split would not change what the model sees.

**4. The errors cluster in plantations that SDPT does not map.**
- **Miss rates:** 32% outside SDPT vs 16% inside (Fisher p = 0.0003, clustered p = 0.0002). The gap shows in every
  model and every state.
- **Their histories:** fewer and smaller cuts, shorter intervals between cuts, faster regrowth. That fits small farm
  woodlots and firewood plots rather than industrial stands.
- **MapBiomas misses them too:** 30 of the 39 plantations its 10 m map calls forest are outside SDPT.
- **Cropin's QC clearings are this kind of plantation:** 431 of the 433 fall outside SDPT (17 within 30 m of a
  polygon), which fits the weaker QC results.

**5. A few "forest" reference points may be mislabelled.**
- **Overall:** 22 MapBiomas natural-forest points lie inside SDPT plantation polygons. They are more likely to be among
  the model's false alarms: 5 of the 67 flagged forest points vs 17 of the other 1,335 (p = 0.003).
- **Points worth checking in high-resolution imagery:**
  - V44803 (Paraná, pine, 4 cuts);
  - V44315 (Santa Catarina, pine, 4 cuts; the MapBiomas 10 m map also calls it plantation);
  - V45429 (Santa Catarina, eucalyptus, 2 cuts; the 10 m map also calls it plantation).

## Files

| Path | Contents |
|---|---|
| `results/misclassified_points.csv` | The 162 MapBiomas points the main model gets wrong (95 missed plantations, 67 forest called plantation), with SDPT species, model score and harvest features |
| `results/val_points_species.csv` | All 1,830 MapBiomas model points with SDPT species and the calls of each model |
| `results/model_predictions_val.csv`, `results/cut_events_val.csv` | Out-of-fold scores and calls, and every detected cut |
| `results/species_analysis.txt` | Every table above |
| `scripts/extract_sdpt.py` | Streams SDPT v2 and writes `public_data/sdpt/` (about 1 minute; needs network access to `gfw2-data.s3.amazonaws.com`) |
| `scripts/notebook_predictions.py` | Notebook code that reproduces the per-point predictions (about 1 minute) |
| `scripts/species_analysis.py` | The analysis |

Run them in that order, from anywhere in the repository. They need numpy, pandas, scipy, shapely, statsmodels and
scikit-learn ≥ 1.6. The first is needed only to rebuild `public_data/sdpt/`, which is committed.

## Caveats and what's next

- **QC clearings are not covered.** Cropin's labels aren't in this environment, and SDPT has no species for 431 of the
  433 clearings anyway.
- **SDPT dates from 2013–14 and maps only large industrial plantations.** A point's species may have changed since,
  and its labels have not been checked here.
- **The model calls come from one cross-validation split,** the same split the notebook uses for its "caught at
  ≤ 5%" figures.
- **Other species sources couldn't be reached from this environment:**
  - IBGE's municipal statistics on pine and eucalyptus area (PEVS, via SIDRA);
  - Global Forest Watch's data API;
  - Zenodo.

  Allowing those hosts would let us cross-check SDPT.
- **Suggested next step:** check species where it matters for Cropin. Spot-check SDPT's species at about 20 points in
  high-resolution imagery. Then look at what the plantations outside SDPT, and the QC plantations, actually are:
  eucalyptus firewood plots, pine woodlots, black wattle. That would show whether a separate small-woodlot detector
  is worth building.
