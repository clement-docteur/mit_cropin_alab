// ============================================================================
// Cropin x MIT – Combined model, Batch M2: MapBiomas land-cover history at every point (land-history idea, midterm)
// Written by Claude, 2026-11-13. Labels are NOT included in this script.
//
// HOW TO RUN (about 1 minute of clicking):
//   1. Paste into code.earthengine.google.com, click Run. The Console prints "points __N__" and the band list.
//      CHECK: the band list should run classification_1985 ... classification_2023. If the asset is not found,
//      tell Claude the error (the asset ID may have changed).
//   2. Tasks tab: click RUN on the one task (default settings). Expected 5–20 min.
//
// DATA: MapBiomas Brazil Collection 9, 30 m Landsat-based yearly land cover, 1985–2023 (the same collection
// as the team's farm-history features). One class code per point and year, read at the point's 30 m pixel.
// The as-of cut (MapBiomas points: up to 2022; QC: up to the year before clearing) is applied later in Python.
//
// FILE PRODUCED (folder cropin_texture): m2_mapbiomas_history.csv
// ============================================================================
var PTS = __PTS__;   // [lon, lat, id, set]
var allPts = ee.FeatureCollection(PTS.map(function(r) {
  return ee.Feature(ee.Geometry.Point([r[0], r[1]]), {id: r[2], set: r[3]});
}));
print('points', allPts.size());
var mb = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1');
print('bands', mb.bandNames());
var out = mb.reduceRegions({collection: allPts, reducer: ee.Reducer.first(), scale: 30, tileScale: 4})
  .map(function(f) { return ee.Feature(null, f.toDictionary()); });
Export.table.toDrive({collection: out, description: 'm2_mapbiomas_history', folder: 'cropin_texture', fileFormat: 'CSV'});
