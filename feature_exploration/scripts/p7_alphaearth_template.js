// ============================================================================
// Cropin x MIT – Feature Exploration, Phase 7, Batch 7: AlphaEarth embeddings (H28 benchmark only)
// Written by Claude, 2026-11-11. Google Satellite Embedding V1 (AlphaEarth Foundations): 64 bands
// (A00–A63), 10 m, one image per year. Used ONLY as a black-box benchmark, never in our model.
// As-of rule: MapBiomas points use 2022 (label window end); QC points use min(cutoff, latest year available).
//
// HOW TO RUN: paste into code.earthengine.google.com, click Run (Console prints "points __N__" and the
// available years), then click RUN on the 1 task. FILE: p7_alphaearth.csv (folder cropin_texture)
// ============================================================================
var PTS = __PTS__;   // [lon, lat, label, id, set, cutoff]
var allPts = ee.FeatureCollection(PTS.map(function(r) {
  return ee.Feature(ee.Geometry.Point([r[0], r[1]]), {id: r[3], set: r[4], cutoff: r[5]});
}));
print('points', allPts.size());
var emb = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');
var years = emb.aggregate_array('system:time_start').map(function(t) { return ee.Date(t).get('year'); }).distinct().sort();
print('years available', years);
var lastYear = ee.Number(years.reduce(ee.Reducer.max()));
var CUTOFFS = __CUTOFFS__;
var out = ee.FeatureCollection(CUTOFFS.map(function(y) {
  var yy = ee.Number(y).min(lastYear);
  var img = emb.filter(ee.Filter.calendarRange(yy, yy, 'year')).mosaic();
  return img.reduceRegions({collection: allPts.filter(ee.Filter.eq('cutoff', y)), reducer: ee.Reducer.first(), scale: 10, tileScale: 4})
    .map(function(f) { return ee.Feature(null, f.toDictionary()).set('emb_year', yy); });
})).flatten();
Export.table.toDrive({collection: out, description: 'p7_alphaearth', folder: 'cropin_texture', fileFormat: 'CSV'});
