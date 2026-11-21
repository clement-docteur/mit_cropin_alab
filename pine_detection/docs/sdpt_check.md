# SDPT check: can WRI's planted-trees map label pine vs eucalyptus? (20 Nov 2026)

Clement Docteur (with Claude). This is the first step of [`../PLAN.md`](../PLAN.md) §2.

## Bottom line

**Yes, partly.** SDPT **v2.0** has a species label for southern Brazil: `Pinus sp.`, `Eucalyptus sp.` and
`Acacia sp.` (black wattle).
- **Source:** the polygons come from Petersen et al. (2016), the original SDPT v1 compilation, with
  **imagery from 2013–2014**.
- **Coverage:** on all 1,831 public MapBiomas points, SDPT gives a species for **262 of the 429
  plantation points (61%)**: 167 pine, 88 eucalyptus and 7 acacia. Another 12 fall in polygons with an
  unknown species, and 155 aren't in any polygon. These are silver labels, not a complete map.
- **Agreement with MapBiomas:** of the 297 points inside an SDPT polygon, **92%** are MapBiomas
  plantations, and only 23 of 1,402 forest points (1.6%) fall inside one.
- **Earth Engine:** SDPT is **not in Earth Engine** (neither the official catalog nor the community
  catalog lists it).

## How to get it

| Route | Status |
|---|---|
| Earth Engine asset | None found |
| Full download, v2.0 (`sdpt_v2.zip`, global File Geodatabase) | 7.7 GB, CC BY 4.0, public S3 link |
| Full download, v2.1 (`sdpt_v21_v09152024_public.gdb.zip`) | 6.9 GB, public S3 link |
| GFW Data API SQL query (`gfw_planted_forests`) | Needs a free GFW API key (an account) |
| **GFW public vector tiles**, version `v20231128` (= v2.0) | Works with no key. A zoom-10 tile is about 40 km across and 0.2–1 MB, and polygon detail is about 10 m |

Tiles of the newest version (`v20239998`, presumably v2.1) come back **empty over southern Brazil**, but
have data over Uruguay. v2.1 says it added "new data in Brazil", so Brazil may be missing from that tile
cache. Use `v20231128` for now.

## Fields (from the GFW Data API)

- **Species:** `sciname`, `simplename`, `sciname1…13`.
- **Type and context:** `simpletype`, `leafstatus`, `leaftype`, `woodtype`, `sizecategory`, `ownership`.
- **Provenance:** `reference`, `imageryyear`, `method`.
- Removal factors and geometry.

In the south, `simplename` is "Wood fiber or timber" and `sciname` carries the species.

## What the tiles show (zoom 10, v2.0, number of polygons)

| Tile (approximate area) | Pinus | Eucalyptus | Acacia | Unknown |
|---|---|---|---|---|
| Caçador / SC midwest | 893 | 227 | — | 17 |
| SC/PR highlands (north of Lages) | 594 | 36 | — | 24 |
| Central RS (near Guaíba) | 200 | 177 | 132 | — |
| Western-central RS | 18 | 304 | 13 | — |

All rows have `reference = Petersen et al. (2016)` and `imageryyear = 2013-2014`. This matches what we
expected: pine in the SC/PR highlands, eucalyptus and black wattle in RS.

## Full overlay (1,831 public MapBiomas points, 412 tiles, 48 MB)

Made with `pine_detection/src/sdpt_overlay.py`. The output is
`data/processed/sdpt_overlay_mapbiomas.csv` (gitignored).

**Species of the SDPT polygon that contains the point:**

| MapBiomas label | State | Pinus | Eucalyptus | Acacia | Unknown | None |
|---|---|---|---|---|---|---|
| Plantation | PR (141) | 65 | 17 | 0 | 8 | 51 |
| Plantation | SC (149) | 63 | 42 | 0 | 3 | 41 |
| Plantation | RS (139) | 39 | 29 | 7 | 1 | 63 |
| **Plantation** | **all (429)** | **167** | **88** | **7** | **12** | **155** |
| Forest | all (1,402) | 16 | 7 | 0 | 0 | 1,379 |

**Allowing up to 100 m to the nearest polygon** (the polygon edges are from 2013–14), 316 plantation
points get a species: 204 pine, 105 eucalyptus and 7 acacia. Only use this looser match as a sensitivity
check, never as the main label.

**What this means for the plan:**
- **Enough for a first model:** about 255 pine/eucalyptus silver labels, with pine outnumbering
  eucalyptus about 2:1. Report balanced accuracy and per-class recall, not plain accuracy.
- **Eucalyptus is thin in PR (17 points).** Results by state will have wide confidence intervals.
- **Acacia (7, all in RS)** is too few to model. Keep it as "other plantation" and leave it out of the
  two-class fit.
- **The 23 forest points inside SDPT polygons** need a look during photo-interpretation. Each one is a
  MapBiomas error, an SDPT error, or a plantation that changed since 2014.
- **The 155 unmatched plantation points** are where we can't get silver labels. They're mostly newer
  plantations, so the gold set should over-sample them.

**Not done yet:**
- joining these labels to the master feature table (`Master Data/`, by MapBiomas point id);
- the QC plantations (run `--bbox` first, as described under Caveats).

## Caveats

- **The labels are 12 years old.** A stand mapped as eucalyptus in 2013–14 has been through about two
  rotations since then. It is usually replanted with the same genus, but it may have changed. Check this
  with the photo-interpretation gold set (PLAN §2.2), and keep `imageryyear` in the label table.
- **Coverage gaps:** plantations created after 2014 are missing, so SDPT labels are biased toward older
  estates. Report the results separately for points with and without SDPT.
- **No accuracy assessment** for this source (WRI's own caution).
- **Acacia is a third class** in RS, which supports a three-class design (PLAN Q3).
- **Privacy:** fetching tiles by location for the **QC polygons** would tell GFW roughly where they are.
  For QC, fetch every tile over PR/SC/RS (a bounding box), not the tiles around each polygon, or use the
  full download.

## Next steps

1. *(Done: see `notebooks/pine_vs_eucalyptus.ipynb`.)* Join the SDPT labels to `Master Data/master_features_v3*.csv`, then test which existing features
   separate pine from eucalyptus, scored on the silver labels.
2. Fetch every tile over PR/SC/RS (`--bbox`), then overlay the QC plantations from the local cache.
3. Photo-interpret a gold set: over-sample the unmatched plantation points and the 23 forest points inside
   SDPT polygons.

## Sources
- [GFW Data API, `gfw_planted_forests`](https://data-api.globalforestwatch.org/dataset/gfw_planted_forests) (metadata, fields, CC BY 4.0)
- [WRI SDPT v2.0](https://www.wri.org/research/spatial-database-planted-trees-sdpt-version-2) · [SDPT v2.1 dataset page](https://datasets.wri.org/datasets/spatial-database-of-planted-trees-sdpt-version-21)
- [GFW / GNW blog on the v2 update](https://www.globalforestwatch.org/blog/data-and-tools/updated-planted-trees-map-near-global-coverage/)
- [Earth Engine community catalog](https://gee-community-catalog.org/) (searched; SDPT not listed)
