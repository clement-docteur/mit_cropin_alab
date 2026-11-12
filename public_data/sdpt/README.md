# Planted-tree species (SDPT v2.0) around the project's points

Extracted by `species_split/scripts/extract_sdpt.py` from the **Spatial Database of Planted Trees, version 2.0**
(World Resources Institute / Global Forest Watch), layer `bra_plant_v2`, downloaded from
`https://gfw2-data.s3.amazonaws.com/plantations/sdpt/sdpt_v2.zip`.

For southern Brazil the layer comes from Transparent World's mapping of large industrial plantations
(Petersen et al. 2016) from 2013–14 imagery. Each polygon is labelled with a species: pine, eucalyptus, acacia, other
or unknown, or young plantation / clearing.

| File | Contents |
|---|---|
| `sdpt_polygons_at_points.geojson` | SDPT polygons whose bounding box comes within ~300 m of a point in `public_data/harvest_cycles/points.csv` (MapBiomas points and QC clearing locations), clipped to ~1 km boxes around those points. Attributes: `OBJECTID`, `vernacName` (species), `sciName`, `originalName`, `sizeCategory`, `reference`, `imageryYear`, `plantedYear` |
| `sdpt_area_by_species.csv` | Number and area (ha) of SDPT polygons by species and state, for polygons within 30 km of a MapBiomas point. Each polygon goes to the state of its nearest MapBiomas point |

Citation: Richter, J., E. Goldman, N. Harris, D. Gibbs, M. Rose, S. Peyer, S. Richardson, and H. Velappan. 2024.
*Spatial Database of Planted Trees (SDPT Version 2.0).* Washington, DC: World Resources Institute.
