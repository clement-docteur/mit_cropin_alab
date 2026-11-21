# Photo-interpretation protocol: pine vs eucalyptus gold labels

Clement Docteur (with Claude), 21 Nov 2026. This is PLAN §2.2. It is fixed **before** anyone labels, so
please don't change the rules mid-way. If something is unclear, write it in `notes` and we'll decide
together afterwards.

## Why we're doing this

SDPT's species labels didn't separate pine from eucalyptus with any feature (`findings.md`). Before
building a model, we need to know **how often SDPT is right**. That is batch 1. Batch 2 then adds labels
where SDPT has none.

## What you get

All files are in `data/processed/gold_sample/` (local only, never committed). Share them through the
team Drive.

| File | Use |
|---|---|
| `gold_batch1.kml`, `gold_batch2.kml` | Open in **Google Earth Pro** (desktop). Each placemark is named by its `sample_id` only |
| `gold_sheet_batch1.csv`, `gold_sheet_batch2.csv` | Copy the file, rename it `..._<your initials>.csv` and fill in one row per point |
| `pine_detection/gee/02_labelling_helper.js` | Optional: Sentinel-2 summer/winter views and the 1985–2024 NBR history for one `sample_id` |
| `gold_key.csv` | **Don't open it until you've finished.** It holds SDPT's and MapBiomas's answers |

Batch 1 has 135 points and batch 2 has 115. At about 1–2 minutes per point, batch 1 takes 3–4 hours per
person.

## Rules

1. **Two people label every point, separately.** Don't discuss points until both sheets are done.
2. **Label the 10 m pixel at the point**, meaning the area within about 15 m. If the point sits on a
   stand edge or a road, label what covers most of that circle and add `edge` to `notes`.
3. **Main label (`species`)** is what grows at the point in the **most recent clear high-resolution image
   from 2019–2023** (our features use those years). Write that image's date in `imagery_date`
   (YYYY-MM).
4. **`species_2014`** is the same question for the clearest image from **2012–2015** (Google Earth Pro's
   historical imagery slider). Leave it blank if there's no usable image. This tells us whether SDPT was
   wrong or the stand changed since.
5. **Saying unsure is fine.** `unsure` with a note is much more useful than a guess.

## Classes (use exactly these words)

| `species` | Meaning |
|---|---|
| `pine` | Planted conifer: *Pinus taeda*, *P. elliottii* |
| `eucalyptus` | Planted *Eucalyptus* (or *Corymbia*) |
| `other_plantation` | Another planted tree crop in regular stands: black wattle (acácia-negra), planted araucaria, erva-mate, etc. Name it in `notes` |
| `clearcut` | A plantation stand that was recently harvested (bare soil, slash, rows of stumps), so the species can't be seen. Use `species_2014` or the NBR history to guess, and say so in `notes` |
| `natural_forest` | Irregular native canopy, including araucaria forest and wild pine invading native vegetation (team rule: wild pine = forest) |
| `not_tree` | Pasture, crop, water, built-up |
| `unsure` | Can't tell. Say why in `notes` |

**`confidence`:** 3 = certain, 2 = probably, 1 = weak guess. Use `unsure` rather than confidence 1 when
you really can't tell.

**`evidence`:** list the cues you used, separated by `;`, from the list below (for example
`colour;crown;rotation`).

## How to tell pine from eucalyptus

| Cue (`evidence` code) | Pine | Eucalyptus |
|---|---|---|
| `colour`: canopy colour at high resolution | **Dark**, blue-green to almost black-green | **Lighter**, grey-green or olive. Often silvery |
| `crown`: crown shape from above | Distinct **star- or cone-shaped** crowns, each with a shadow on one side. Rough, "fuzzy" texture | **Smooth, cotton-like** canopy. Individual crowns are hard to see once the canopy closes |
| `rows`: row pattern | Rows visible, often with **gaps from thinning** (every 4th–5th row removed) | Very regular rows, usually **not thinned** |
| `trunk`: Street View or oblique view | Dark, rough bark. Dead lower branches | **Smooth pale or white trunks**, peeling bark. Bare stems up to the canopy |
| `rotation`: historical slider / NBR history | Cut every **12–25 years**, with 2–3 thinnings in between (partial drops) | **Clear-cut every 6–8 years**, full regrowth in 2–3 years, no thinning |
| `region`: weak prior only, never the only cue | More common in the SC/PR highlands (Serra Catarinense, Campos Gerais, Caçador) | More common in RS and the lowlands. Large pulp estates (Guaíba, Telêmaco Borba) |
| `streetview`: road signs or company signs | Timber or resin companies | Pulp companies |

Black wattle (`other_plantation`, mostly RS) looks like young eucalyptus but has a **darker, finer, very
dense** canopy and very short rotations. If it looks like eucalyptus but feels off, write
`unsure` + `wattle?`.

## After labelling

1. Put both sheets in `data/processed/gold_sample/labels/`.
2. A new section of `notebooks/pine_vs_eucalyptus.ipynb` (still to be written) will:
   - check that the sheets are filled in properly;
   - compute agreement (Cohen's κ) on `species`;
   - list disagreements for a joint review. The agreed label goes in `species_gold`;
   - compare `species_gold` with SDPT and with MapBiomas, by stratum. Only at this point is `gold_key.csv`
     opened.
3. **Gold label** = both interpreters agree, or a disagreement resolved jointly, at confidence ≥ 2.
   Everything else is kept but flagged.

## What we'll report from batch 1

- **SDPT accuracy by species:** of the 50 SDPT-pine and 50 SDPT-eucalyptus points, the share that are
  that species now, and the share that were in 2012–15. Each with a 95% CI (with 50 points, about
  ±14 points).
- **What the 35 MapBiomas-forest points inside SDPT polygons really are.**
- **Decision rule, set now:** if SDPT matches our 2012–15 label at ≥ 85% for both species, keep it as
  silver training labels. Otherwise train only on gold labels.
