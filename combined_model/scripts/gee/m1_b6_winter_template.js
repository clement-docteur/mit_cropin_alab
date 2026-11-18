// ============================================================================
// Cropin x MIT – Combined model, Batch M1: Sentinel-2 B6 (red edge, 740 nm) monthly medians, Apr–Sep
// Needed for Meera's "winter red edge" feature (Findings deck 12 Nov). Written by Claude, 2026-11-13.
//
// HOW TO RUN (about 1 minute of clicking):
//   1. Paste into code.earthengine.google.com, click Run. Console prints "points __N__".
//   2. Tasks tab: click RUN on both tasks (default settings). Expected 10–40 min each.
//
// TIME WINDOW (as-of rule, same as Wave 2): the 3 calendar years ending at each point's cutoff year
// (MapBiomas: 2020–2022; QC: the 3 years before the estimated clearing year). Only April–September.
// Same cloud mask as Wave 2 (Cloud Score+ cs >= 0.6). Labels are NOT included in this script.
//
// FILES PRODUCED (folder cropin_texture): m1_b6_winter_mb.csv, m1_b6_winter_qc.csv
// ============================================================================
var PTS = __PTS__;   // [lon, lat, id, set, cutoff]
var allPts = ee.FeatureCollection(PTS.map(function(r) {
  return ee.Feature(ee.Geometry.Point([r[0], r[1]]), {id: r[2], set: r[3], cutoff: r[4]});
}));
print('points', allPts.size());
var CUTOFFS = {mb: __CUT_MB__, qc: __CUT_QC__};
var CS = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
function strip(fc) { return fc.map(function(f) { return ee.Feature(null, f.toDictionary()); }); }
function b6winter(pts, y) {
  var reg = pts.geometry().bounds();
  var months = [];
  [y - 2, y - 1, y].forEach(function(yr) { for (var m = 4; m <= 9; m++) { months.push([yr, m]); } });
  return ee.FeatureCollection(months.map(function(ym) {
    var start = ee.Date.fromYMD(ym[0], ym[1], 1);
    var c = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(reg)
      .filterDate(start, start.advance(1, 'month')).linkCollection(CS, ['cs'])
      .map(function(im) { return im.updateMask(im.select('cs').gte(0.6)); }).select(['B6']);
    var empty = ee.Image.constant([0, 0]).rename(['B6', 'n_clear']).updateMask(0).toFloat();
    var img = ee.Image(ee.Algorithms.If(c.size().gt(0),
      c.median().addBands(c.count().rename('n_clear')).toFloat(), empty));
    return strip(img.reduceRegions({collection: pts, reducer: ee.Reducer.first(), scale: 10, tileScale: 4}))
      .map(function(f) { return f.set('month', start.format('YYYY-MM')); });
  })).flatten();
}
function byCutoff(setName, fn) {
  return ee.FeatureCollection(CUTOFFS[setName].map(function(y) {
    return fn(allPts.filter(ee.Filter.and(ee.Filter.eq('set', setName), ee.Filter.eq('cutoff', y))), y);
  })).flatten();
}
Export.table.toDrive({collection: byCutoff('mb', b6winter), description: 'm1_b6_winter_mb', folder: 'cropin_texture', fileFormat: 'CSV'});
Export.table.toDrive({collection: byCutoff('qc', b6winter), description: 'm1_b6_winter_qc', folder: 'cropin_texture', fileFormat: 'CSV'});
