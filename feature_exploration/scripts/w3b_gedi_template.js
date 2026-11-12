// ============================================================================
// Cropin x MIT – Feature Exploration, Wave 3, Batch 3b: GEDI re-export
// Written by Claude, 2026-11-06. In batch 3 the GEDI mean/SD columns were dropped from the CSV
// because the first rows had no shots (null values); Earth Engine CSV exports keep only the first
// rows' columns. Fix: every missing value is written as -9999. Also adds a 250 m radius version (same row).
// HOW TO RUN: paste, click Run (Console prints "points __N__"), click RUN on the 1 task.
// FILE: w3_gedi_v2.csv (folder cropin_texture)
// ============================================================================
var PTS = __PTS__;   // [lon, lat, label, id, set, cutoff, season_start]
var allPts = ee.FeatureCollection(PTS.map(function(r) {
  return ee.Feature(ee.Geometry.Point([r[0], r[1]]), {label: r[2], id: r[3], set: r[4], cutoff: r[5], season_start: r[6]});
}));
var region = allPts.geometry().bounds();
print('points', allPts.size());
var CUTOFFS = {mb: __CUT_MB__, qc: __CUT_QC__};
function strip(fc) { return fc.map(function(f) { return ee.Feature(null, f.toDictionary()); }); }
function byCutoff(setName, fn) {
  return ee.FeatureCollection(CUTOFFS[setName].map(function(y) {
    return fn(allPts.filter(ee.Filter.and(ee.Filter.eq('set', setName), ee.Filter.eq('cutoff', y))), y);
  })).flatten();
}
function sample(img, pts, scale, ts) {
  return strip(img.reduceRegions({collection: pts, reducer: ee.Reducer.first(), scale: scale, tileScale: ts || 4}));
}

var KEYS = ['rh98_mean', 'rh98_stdDev', 'rh98_count', 'cover_mean', 'cover_stdDev', 'cover_count',
            'fhd_normal_mean', 'fhd_normal_stdDev', 'fhd_normal_count', 'pai_mean', 'pai_stdDev', 'pai_count'];
function fill(f, suffix) {
  var d = ee.Dictionary({id: f.get('id'), set: f.get('set'), label: f.get('label'), cutoff: f.get('cutoff')});
  KEYS.forEach(function(k) {
    d = d.set(k + suffix, ee.Algorithms.If(ee.Algorithms.IsEqual(f.get(k), null), -9999, f.get(k)));
  });
  return ee.Feature(null, d);
}
function gedi(pts, y) {
  var s = ee.Date.fromYMD(y - 2, 1, 1), e = ee.Date.fromYMD(y + 1, 1, 1);
  var a = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').filterBounds(pts.geometry().bounds()).filterDate(s, e)
    .map(function(im) { return im.updateMask(im.select('quality_flag').eq(1).and(im.select('degrade_flag').eq(0))).select('rh98'); });
  var b = ee.ImageCollection('LARSE/GEDI/GEDI02_B_002_MONTHLY').filterBounds(pts.geometry().bounds()).filterDate(s, e)
    .map(function(im) { return im.updateMask(im.select('l2b_quality_flag').eq(1).and(im.select('degrade_flag').eq(0))).select(['cover', 'fhd_normal', 'pai']); });
  var img = a.median().addBands(b.median());
  var red = ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true).combine(ee.Reducer.count(), '', true);
  var r100 = img.reduceRegions({collection: pts.map(function(f) { return f.buffer(100); }), reducer: red, scale: 25, tileScale: 8})
    .map(function(f) { return fill(f, '_r100'); });
  var r250 = img.reduceRegions({collection: pts.map(function(f) { return f.buffer(250); }), reducer: red, scale: 25, tileScale: 8})
    .map(function(f) { return fill(f, '_r250'); });
  // one row per point (separate rows would lose the 250 m columns in the CSV)
  return ee.Join.inner().apply(r100, r250, ee.Filter.equals({leftField: 'id', rightField: 'id'})).map(function(j) {
    return ee.Feature(null, ee.Feature(j.get('primary')).toDictionary().combine(ee.Feature(j.get('secondary')).toDictionary()));
  });
}
Export.table.toDrive({collection: byCutoff('mb', gedi).merge(byCutoff('qc', gedi)), description: 'w3_gedi_v2', folder: 'cropin_texture', fileFormat: 'CSV'});
