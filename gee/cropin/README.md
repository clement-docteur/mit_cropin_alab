# Cropin's own Earth Engine scripts

Shared by Uddesh and Deepak (Cropin) by email on **2026-10-28**, as GEE snapshot links. We had
asked for the SAG and CCDC code at the 10/21 meeting. The scripts are copied **verbatim**: only
the header block with the snapshot link was added. **No CCDC code has arrived yet; we still need
to ask for it.**

| Script | Snapshot |
|---|---|
| `cropin_forest_layer.js` | https://code.earthengine.google.com/6a85964d7c9cc0982dfd60c98e41cb9a |
| `cropin_multiyear_SAG_plantation.js` | https://code.earthengine.google.com/96c128110c3ee110203d96734a1083c6 |
| `cropin_multipoint_SAG_timeseries_explorer.js` | https://code.earthengine.google.com/d243b59bbc051a59d96980776bc0bd9a |

---

## 1. `cropin_forest_layer.js`: how Cropin decides "this is forest"

This is very likely the layer that produced the alerts in our QC set. It works in two stages.

**Stage 1: a vote between four forest maps for 2020.** Each map says forest (1) or not (0)
for every pixel. The votes are weighted and averaged into a score between 0 and 1:

| Map | Counts as forest when | Weight |
|---|---|---|
| Hansen Global Forest Change v1.12 | tree cover in 2000 ≥ 30% and no loss recorded 2001–2020 | 2 |
| JRC Global Forest Cover 2020 (V2) | value > 0.5 | 2 |
| MapBiomas Collection 10 (30 m) | class 3, forest formation | 3 |
| JRC Tropical Moist Forest (2024 release), Dec 2020 | change class ≤ 2 or = 4 | 3 |

`score = (2·Hansen + 2·JRC + 3·MapBiomas + 3·TMF) / 10`

**Stage 2: three filters, all required.**
1. **Not plantation in MapBiomas 10 m.** The pixel must not be class 9 in either MapBiomas
   10 m Collection 2 or the 10 m Sentinel-2 beta.
2. **Forest in the MapBiomas 10 m S2 beta.** The pixel must be class 3 there.
3. **Tall enough.** Meta's 1 m canopy height must be ≥ 4 m. The code comment says "≥ 1 m",
   but the code uses 4.

**Forest = score ≥ 0.75** after those filters.

**What this means for us.** By design, the layer removes anything MapBiomas 10 m already calls
plantation. So MapBiomas 10 m should "miss" nearly all QC plantations, and our job is to add
information *on top of* this layer. ⚠️ Our export does not fully match this design: at polygon
level only 169 of 310 QC plantations and 94 of 123 QC forests pass the final flag. See
`notebooks/02_cropin_layers_and_SAG.ipynb`. The alerts may come from a different version or
year of the layer. That is a question for Cropin.

## 2. `cropin_multiyear_SAG_plantation.js`: SAG plantation probability

**The idea.** From October to January (southern summer), plantations look a darker, flatter
green than native forest in Sentinel-2. Uddesh measured about a 10° difference in SAG.

**What SAG is.** Plot a pixel's blue, green and red reflectance against wavelength (0.49, 0.56,
0.66 µm). That gives three points. SAG (Spectral Angle at Green) is the angle at the green point
between the line to blue and the line to red:
- a sharp green peak (bright, "lighter" green) gives a smaller angle;
- a flat, dark spectrum gives an angle closer to 180°.

Only dark pixels count: blue, green and red reflectance must all be ≤ 0.05.

**The Bayesian update, step by step.** It runs separately for each pixel and each season
(Oct 1 to Jan 30, seasons 2019/20 to 2025/26):
1. Start at 50/50: P(plantation) = 0.5.
2. For each clear image (Cloud Score+ ≥ 0.6), read the pixel's SAG and ask how likely that
   value is for each class. Plantation SAG is assumed to follow a bell curve centred at 160°
   (SD 9°). Non-plantation is centred at 150° (SD 9°).
3. Bayes' rule: new P = (plantation likelihood × old P) / (plantation likelihood × old P +
   non-plantation likelihood × (1 − old P)). Each image nudges the probability towards the
   class whose curve fits better.
4. Clamp P to 0.1–0.9 after every image, so no single season becomes certain. If the pixel is
   masked on a date, keep the previous P.
5. The season's last P is the season's posterior.

The **7-season mean posterior** is then mapped. **Plantation = mean ≥ 0.8**, shown only where
Meta height ≥ 4 m. MapBiomas 10 m plantation is overlaid for comparison.

**Quirks to know.**
- **Cloud mask.** `addGreenAngle()` ends with `.mask(band_value_mask)`, which *replaces* the
  Cloud Score+ mask instead of adding to it. Dark cloud shadows can then pass. Our script 03
  uses `updateMask` instead.
- **Fixed curves.** The bell-curve centres and widths (160/150, SD 9) are set by hand, not
  fitted to labelled data. With equal SDs, each image's evidence depends only on whether SAG is
  above or below 155°, and how far.
- **Window end.** The season ends on Jan 30, because the date filter's end date is exclusive.
- **Known failure.** A dark native pine (likely *Araucaria*) also looks dark green and gets
  called plantation. Uddesh thinks this is more than 2–3% of cases.
- **Never validated.** It has not been tested on labelled data, including our QC polygons,
  before notebook 02.

## 3. `cropin_multipoint_SAG_timeseries_explorer.js`: click-and-compare tool

An interactive inspector. Click points on the map, for example one known plantation and one
known forest, and it charts SAG over time for each:
- Sentinel-2 with Cloud Score+ masking and the same dark-pixel rule;
- 10-day median composites from May 2022 to May 2025.

Use it for eyeballing: do plantations sit higher than forest, and in which months? It is also a
quick way to check the **winter (Jun–Aug) vs summer (Oct–Jan)** idea, which Deepak said nobody
had looked at. It produces no export.

---

How we apply these to the 433 QC polygons is in `gee/03_cropin_layers_and_SAG_qc_export.js`.
The SAG posterior is rebuilt in Python in notebook 02.
