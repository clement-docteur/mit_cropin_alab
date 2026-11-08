// ============================================================================
// Cropin x MIT – Feature Exploration, Wave 1, Batch 1c: LandTrendr + CCDC refit at the as-of cutoff
// Written by Claude, 2026-11-06. Fixes reviewer item 7: the first fits used data up to 2025, so
// 2023–2025 shaped MapBiomas features (labels end 2022). Here each series ENDS at the point's cutoff:
// MapBiomas 2022; QC = year before the estimated clearing (only QC points with cutoff < 2025).
//
// HOW TO RUN: paste into code.earthengine.google.com, click Run (Console prints "points __N__"),
// then click RUN on both tasks. Expected time 10–60 min each.
// FILES: w1c_landtrendr_cut.csv, w1c_ccdc_cut.csv (folder cropin_texture)
// Settings are identical to batch 1 (w1_batch1_gee.js) except the series end year.
// ============================================================================
var PTS = __PTS__;   // [lon, lat, label, id, set, cutoff]
var allPts = ee.FeatureCollection(PTS.map(function(r) {
  return ee.Feature(ee.Geometry.Point([r[0], r[1]]), {label: r[2], id: r[3], set: r[4], cutoff: r[5]});
}));
var region = allPts.geometry().bounds();
print('points', allPts.size());

function prep(img, bands) {
  var qa = img.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 1).eq(0).and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0)).and(qa.bitwiseAnd(1 << 5).eq(0));
  var sr = img.select(bands, ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).multiply(0.0000275).add(-0.2);
  return sr.addBands([sr.normalizedDifference(['nir', 'red']).rename('NDVI'),
                      sr.normalizedDifference(['nir', 'swir2']).rename('NBR')])
    .updateMask(clear).copyProperties(img, ['system:time_start']);
}
var L57 = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'];
var L89 = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];
function landsat(start, end) {
  function get(id, b) {
    return ee.ImageCollection(id).filterBounds(region).filterDate(start, end).map(function(i) { return prep(i, b); });
  }
  return get('LANDSAT/LT05/C02/T1_L2', L57).merge(get('LANDSAT/LE07/C02/T1_L2', L57))
    .merge(get('LANDSAT/LC08/C02/T1_L2', L89)).merge(get('LANDSAT/LC09/C02/T1_L2', L89)).sort('system:time_start');
}
function annualNBR(y) {
  y = ee.Number(y);
  var c = landsat(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y.add(1), 1, 1));
  var img = ee.Image(ee.Algorithms.If(c.size().gt(0), c.select('NBR').median(),
                                      ee.Image.constant(0).rename('NBR').updateMask(0).toFloat()));
  return img.multiply(-1000).toShort().rename('NBR').set('system:time_start', ee.Date.fromYMD(y, 7, 1).millis());
}
function ltFor(pts, last) {
  var col = ee.ImageCollection(ee.List.sequence(1985, last).map(annualNBR));
  var lt = ee.Algorithms.TemporalSegmentation.LandTrendr({
    timeSeries: col, maxSegments: 6, spikeThreshold: 0.9, vertexCountOvershoot: 3,
    preventOneYearRecovery: true, recoveryThreshold: 0.25, pvalThreshold: 0.05,
    bestModelProportion: 0.75, minObservationsNeeded: 6}).select('LandTrendr');
  return lt.reduceRegions({collection: pts, reducer: ee.Reducer.first(), scale: 30, tileScale: 8})
    .map(function(f) { return ee.Feature(null, f.toDictionary()); });
}
function ccdcFor(pts, last) {
  var ccdc = ee.Algorithms.TemporalSegmentation.Ccdc({
    collection: landsat('2000-01-01', (last + 1) + '-01-01').select(['green', 'red', 'nir', 'swir1', 'swir2', 'NDVI', 'NBR']),
    breakpointBands: ['green', 'red', 'nir', 'swir1', 'swir2'], tmaskBands: ['green', 'swir2'],
    minObservations: 6, chiSquareProbability: 0.99, minNumOfYearsScaler: 1.33, dateFormat: 1,
    lambda: 0.002, maxIterations: 10000});
  return ccdc.select(['tStart', 'tEnd', 'tBreak', 'numObs', 'changeProb', 'NDVI_coefs', 'NDVI_rmse',
                      'NDVI_magnitude', 'NBR_magnitude', 'swir1_magnitude'])
    .reduceRegions({collection: pts, reducer: ee.Reducer.first(), scale: 30, tileScale: 16})
    .map(function(f) { return ee.Feature(null, f.toDictionary()); });
}
var CUTOFFS = __CUTOFFS__;
var ltAll = ee.FeatureCollection(CUTOFFS.map(function(y) { return ltFor(allPts.filter(ee.Filter.eq('cutoff', y)), y); })).flatten();
var ccAll = ee.FeatureCollection(CUTOFFS.map(function(y) { return ccdcFor(allPts.filter(ee.Filter.eq('cutoff', y)), y); })).flatten();
Export.table.toDrive({collection: ltAll, description: 'w1c_landtrendr_cut', folder: 'cropin_texture', fileFormat: 'CSV'});
Export.table.toDrive({collection: ccAll, description: 'w1c_ccdc_cut', folder: 'cropin_texture', fileFormat: 'CSV'});
