# P02 findings: pine vs eucalyptus on SDPT labels (20 Nov 2026)

Clement Docteur (with Claude). Source: `notebooks/P02_species_univariate.ipynb`, built by
`notebooks/build_P02.py`.
- Figures: `figures/P02_univariate_auc.png`, `figures/P02_rotation_distributions.png`.
- Tables: `results/P02_univariate.csv`, `results/P02_families_cv.csv`.

**Status:** the notebook ran on 21 Nov 2026 and is saved with its outputs. The numbers are identical to
the first run. On 20 Nov, Windows Application Control blocked a scikit-learn DLL for a while. If that
happens again, wait and rerun.

## Bottom line

**Against SDPT's species labels, almost nothing separates pine from eucalyptus.**
- **Labels too coarse:** latitude is the best "feature", so SDPT seems right by region but not stand by
  stand.
- **Gold set now critical:** the photo-interpreted gold set is the critical path, and its first job is to
  measure how often SDPT's species is right.

## Numbers

**Sample:** 245 MapBiomas plantation points, development split (85 eucalyptus, 160 pine), labelled by
being inside an SDPT v2.0 polygon. The lockbox split (17 eucalyptus, 39 pine) is untouched.

| Check | Result |
|---|---|
| Best single feature | Latitude: AUC 0.64 [0.57–0.72], Holm p = 0.008. Eucalyptus lies further south. It is the only feature significant after correction |
| Best satellite features | Share of years below canopy 0.60 [0.53–0.68]. B8A narrow NIR in winter 0.57 [0.49–0.64] and B6 red-edge in winter 0.56 [0.49–0.64], both higher in eucalyptus as expected for broadleaf vs needles |
| New rotation features | AUC 0.50–0.52, unchanged with a clear-cut threshold of 0.15 or 0.25. Median years between clear-cuts is **11 for both species** |
| Feature families, logistic regression, spatially blocked CV (20 blocks, seeds 0–19) | History + rotation 0.57. Geography 0.54. Spectral 0.44. Structure 0.40. AlphaEarth 0.48. Everything together 0.51 |
| Random forest (seeds 0–4) | Geography 0.51. All features 0.49. All + AlphaEarth 0.47 |
| Within each state (top features) | AUCs of 0.5–0.74 that rise and fall from state to state. PR has only 17 eucalyptus points |

AUCs below 0.5 in cross-validation mean the model learned nothing that carries over to the left-out
regions. They don't mean a signal in the reverse direction.

## Why we think the labels are the problem

1. **Pine and eucalyptus do differ in satellite data.** The IPEF national map and a UDESC thesis in Santa
   Catarina both separate them with Sentinel-2. Here the expected NIR/red-edge direction appears, but only
   weakly.
2. **Eucalyptus should be cut about every 6–8 years,** and pine about every 12–25. Identical rotation
   lengths point to labels that are wrong for individual stands.
3. **SDPT's southern Brazil polygons are 2013–14 data from one compilation,** with no accuracy assessment.
   35 MapBiomas *forest* points also fall inside SDPT polygons.

## Alternative explanations (not ruled out)

- **Weak features:** the master-table features were chosen to separate plantation from forest, not species
  from each other. Species-specific ones haven't been tested, such as the full band × month grid (the
  gee/05 exports aren't on this machine), texture (row spacing, crown shape) or red-edge ratios.
- **Too-coarse NBR series:** the yearly median NBR comes from about 6 Landsat images a year, so a fast
  eucalyptus regrowth (2–3 years) may hide some clear-cuts.

## Next steps

1. **Gold labels:** photo-interpret about 50–100 SDPT-labelled points to measure SDPT's accuracy, then add
   unmatched points to reach about 300 (PLAN §2.2).
2. If SDPT turns out to be mostly right, test species-specific features: band × month grid, texture,
   red-edge ratios.
3. Tell Cropin, by 12/1, that the public species labels don't hold up at stand level, and ask again
   whether they or their clients have species-labelled stands (PLAN Q2).
