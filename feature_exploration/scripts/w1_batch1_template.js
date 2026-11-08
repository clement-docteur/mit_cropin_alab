// ============================================================================
// Cropin x MIT – Feature Exploration, Wave 1, Batch 1
// Plan: Feature Exploration/00 Workflow Plan.md (v4.1). Written by Claude, 2026-11-05.
//
// HOW TO RUN (about 3 minutes of clicking):
//   1. Paste this whole script into code.earthengine.google.com and click Run.
//      The Console should print: "MapBiomas points __N_MB__" and "QC points __N_QC__".
//   2. Open the Tasks tab. Six tasks appear. Click RUN on each (default settings).
//   3. Files land in Drive folder "cropin_texture" and sync to the Mac.
//      Expected time: 10–60 min per task. w1_ccdc is the slowest and the most
//      likely to fail; if it does, tell Claude (the other five do not depend on it).
//
// FILES PRODUCED (folder cropin_texture):
//   w1_landsat_annual.csv  Family D: yearly Landsat NDVI / NBR / SWIR1 per point, 1985–2025
//   w1_landtrendr.csv      Family D: LandTrendr segmentation of yearly NBR (vertices + fitted values)
//   w1_ccdc.csv            Family D: CCDC breaks and harmonic fit, Landsat 2000–2025
//   w1_hansen.csv          Family D (H6): Hansen loss years within 150 m (neighbours cleared together)
//   w1_context.csv         Alert-like subset + Family G: tree cover around the point, patch size,
//                          distance to non-tree, elevation, slope
//   w1_texture_extra.csv   Family C: SNIC segment stats, directional contrast (H16 test),
//                          texture on red (B4) and SWIR2 (B12)
// ============================================================================

var MB_RAW = __MB_RAW__;
var QC_RAW = __QC_RAW__;
function toFC(raw, set) {
  return ee.FeatureCollection(raw.map(function(r) {
    return ee.Feature(ee.Geometry.Point([r[0], r[1]]), {label: r[2], id: r[3], set: set});
  }));
}
var mbPts = toFC(MB_RAW, 'mb'), qcPts = toFC(QC_RAW, 'qc'), allPts = mbPts.merge(qcPts);
var region = allPts.geometry().bounds();
print('MapBiomas points', mbPts.size(), 'QC points', qcPts.size());

// ---------------------------------------------------------------- Landsat C2 L2
function prep(img, bands) {
  var qa = img.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 1).eq(0).and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0)).and(qa.bitwiseAnd(1 << 5).eq(0));
  var sr = img.select(bands, ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
    .multiply(0.0000275).add(-0.2);
  var ndvi = sr.normalizedDifference(['nir', 'red']).rename('NDVI');
  var nbr = sr.normalizedDifference(['nir', 'swir2']).rename('NBR');
  return sr.addBands([ndvi, nbr]).updateMask(clear)
    .copyProperties(img, ['system:time_start']);
}
var L57 = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'];
var L89 = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];
function landsat(start, end) {
  function get(id, b) {
    return ee.ImageCollection(id).filterBounds(region).filterDate(start, end)
      .map(function(i) { return prep(i, b); });
  }
  return get('LANDSAT/LT05/C02/T1_L2', L57).merge(get('LANDSAT/LE07/C02/T1_L2', L57))
    .merge(get('LANDSAT/LC08/C02/T1_L2', L89)).merge(get('LANDSAT/LC09/C02/T1_L2', L89))
    .sort('system:time_start');
}
var YEARS = ee.List.sequence(1985, 2025);
function annual(y) {
  y = ee.Number(y);
  var c = landsat(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y.add(1), 1, 1));
  var empty = ee.Image.constant([0, 0, 0, 0, 0]).rename(['NDVI', 'NBR', 'swir1', 'NBR_min', 'n_clear'])
    .updateMask(0).toFloat();
  var img = ee.Image(ee.Algorithms.If(c.size().gt(0),
    c.select(['NDVI', 'NBR', 'swir1']).median()
      .addBands(c.select('NBR').min().rename('NBR_min'))
      .addBands(c.select('NDVI').count().rename('n_clear')).toFloat(),
    empty));
  return img.set('year', y).set('system:time_start', ee.Date.fromYMD(y, 7, 1).millis());
}
var annualCol = ee.ImageCollection(YEARS.map(annual));

// ---------- 1. yearly Landsat values per point ----------
var annualTable = annualCol.map(function(img) {
  return img.reduceRegions({collection: allPts, reducer: ee.Reducer.first(), scale: 30, tileScale: 4})
    .map(function(f) { return ee.Feature(null, f.toDictionary()).set('year', img.get('year')); });
}).flatten();
Export.table.toDrive({collection: annualTable, description: 'w1_landsat_annual',
  folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 2. LandTrendr on yearly NBR (disturbance = positive, so NBR x -1000) ----------
var ltInput = annualCol.map(function(img) {
  return img.select('NBR').multiply(-1000).toShort().rename('NBR')
    .copyProperties(img, ['system:time_start']);
});
var lt = ee.Algorithms.TemporalSegmentation.LandTrendr({
  timeSeries: ltInput, maxSegments: 6, spikeThreshold: 0.9, vertexCountOvershoot: 3,
  preventOneYearRecovery: true, recoveryThreshold: 0.25, pvalThreshold: 0.05,
  bestModelProportion: 0.75, minObservationsNeeded: 6}).select('LandTrendr');
// Exported as the raw array (rows: year, observed, fitted, is_vertex); Claude parses it in Python.
Export.table.toDrive({
  collection: lt.reduceRegions({collection: allPts, reducer: ee.Reducer.first(), scale: 30, tileScale: 8})
    .map(function(f) { return ee.Feature(null, f.toDictionary()); }),
  description: 'w1_landtrendr', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 3. CCDC (Landsat 2000–2025) ----------
var ccdcIn = landsat('2000-01-01', '2026-01-01').select(['green', 'red', 'nir', 'swir1', 'swir2', 'NDVI', 'NBR']);
var ccdc = ee.Algorithms.TemporalSegmentation.Ccdc({
  collection: ccdcIn, breakpointBands: ['green', 'red', 'nir', 'swir1', 'swir2'],
  tmaskBands: ['green', 'swir2'], minObservations: 6, chiSquareProbability: 0.99,
  minNumOfYearsScaler: 1.33, dateFormat: 1, lambda: 0.002, maxIterations: 10000});
var ccdcOut = ccdc.select(['tStart', 'tEnd', 'tBreak', 'numObs', 'changeProb',
  'NDVI_coefs', 'NDVI_rmse', 'NDVI_magnitude', 'NBR_magnitude', 'swir1_magnitude']);
Export.table.toDrive({
  collection: ccdcOut.reduceRegions({collection: allPts, reducer: ee.Reducer.first(), scale: 30, tileScale: 16})
    .map(function(f) { return ee.Feature(null, f.toDictionary()); }),
  description: 'w1_ccdc', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 4. Hansen loss years within 150 m (H6: neighbours cleared in the same year) ----------
var hansen = ee.Image('UMD/hansen/global_forest_change_2025_v1_13');
var lossyear = hansen.select('lossyear');   // 0 = no loss; 1–25 = loss in 2001–2025
var hTable = lossyear.reduceRegions({collection: allPts.map(function(f) { return f.buffer(150); }),
    reducer: ee.Reducer.frequencyHistogram(), scale: 30, tileScale: 4})
  .map(function(f) { return ee.Feature(null, f.toDictionary()); });
var tc = hansen.select('treecover2000').reduceRegions({collection: allPts, reducer: ee.Reducer.first(), scale: 30})
  .map(function(f) { return ee.Feature(null, {id: f.get('id'), set: f.get('set'), treecover2000: f.get('first')}); });
Export.table.toDrive({collection: hTable.merge(tc), description: 'w1_hansen',
  folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 5. Context: tree cover around the point, patch size, edge distance, terrain ----------
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');   // 2021, 10 m
var tree = wc.eq(10);
var ctx = ee.Image.cat([
  tree.rename('wc_tree'),
  tree.selfMask().connectedPixelCount({maxSize: 1024, eightConnected: true}).unmask(0).rename('wc_patch_px'),
  tree.not().fastDistanceTransform(256).sqrt().multiply(10).rename('wc_dist_nontree_m'),
  tree.reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: ee.Kernel.circle(5)}).rename('wc_tree_frac_50m'),
  tree.reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: ee.Kernel.circle(10)}).rename('wc_tree_frac_100m'),
  tree.reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: ee.Kernel.circle(25)}).rename('wc_tree_frac_250m')
]);
var dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic()
  .setDefaultProjection('EPSG:4326', null, 30);
ctx = ctx.addBands(dem.rename('elev')).addBands(ee.Terrain.slope(dem).rename('slope'));
Export.table.toDrive({
  collection: ctx.sampleRegions({collection: allPts, scale: 10, geometries: false, tileScale: 4}),
  description: 'w1_context', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 6. Family C extras on Sentinel-2 summer composites ----------
var CS = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
function s2summer(y, reg) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(reg)
    .filterDate(y + '-12-01', (y + 1) + '-04-01').linkCollection(CS, ['cs'])
    .map(function(i) { return i.updateMask(i.select('cs').gte(0.6)); })
    .select(['B4', 'B8', 'B11', 'B12']).median();
}
function extras(img) {
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var x = img.divide(10000).addBands(ndvi);
  // SNIC segments (seed spacing 10 px = 100 m)
  var snic = ee.Algorithms.Image.Segmentation.SNIC({image: x.select(['B4', 'B8', 'B11']),
    size: 10, compactness: 1, connectivity: 8});
  var cl = snic.select('clusters');
  var segSd = x.select(['B8', 'B11', 'NDVI']).addBands(cl)
    .reduceConnectedComponents({reducer: ee.Reducer.stdDev(), labelBand: 'clusters', maxSize: 1024})
    .rename(['seg_sd_B8', 'seg_sd_B11', 'seg_sd_NDVI']);
  var segN = cl.connectedPixelCount({maxSize: 1024, eightConnected: true}).rename('seg_px');
  // directional contrast (H16 rows test): mean squared neighbour difference in 4 directions, 5x5 window
  var b8 = x.select('B8');
  var dirs = {d0: [[0, 0, 0], [0, -1, 1], [0, 0, 0]], d45: [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
              d90: [[0, 1, 0], [0, -1, 0], [0, 0, 0]], d135: [[1, 0, 0], [0, -1, 0], [0, 0, 0]]};
  var dc = ee.Image([]);
  Object.keys(dirs).forEach(function(k) {
    var d = b8.convolve(ee.Kernel.fixed(3, 3, dirs[k], -1, -1, false)).pow(2)
      .reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: ee.Kernel.square(2)});
    dc = dc.addBands(d.rename('dirc_B8_' + k));
  });
  // GLCM texture on red and SWIR2
  var grey = ee.Image.cat([
    img.select('B4').unitScale(0, 2000).multiply(255).clamp(0, 255).toByte().rename('B4'),
    img.select('B12').unitScale(0, 3000).multiply(255).clamp(0, 255).toByte().rename('B12')]);
  var tex = ee.Image([]);
  [1, 2, 3].forEach(function(r) {
    var g = grey.glcmTexture({size: r});
    ['B4', 'B12'].forEach(function(b) {
      ['contrast', 'ent', 'idm', 'asm', 'corr', 'var'].forEach(function(m) {
        tex = tex.addBands(g.select(b + '_' + m).rename(b + '_' + m + '_w' + (2 * r + 1)));
      });
    });
  });
  return segSd.addBands(segN).addBands(dc).addBands(tex);
}
var mbX = extras(s2summer(2021, mbPts.geometry().bounds()))
  .sampleRegions({collection: mbPts, scale: 10, geometries: false, tileScale: 8})
  .map(function(f) { return f.set('season_start', 2021); });
var qcX = ee.FeatureCollection([2022, 2023, 2024, 2025].map(function(y) {
  return extras(s2summer(y, qcPts.geometry().bounds()))
    .sampleRegions({collection: qcPts, scale: 10, geometries: false, tileScale: 8})
    .map(function(f) { return f.set('season_start', y); });
})).flatten();
Export.table.toDrive({collection: mbX.merge(qcX), description: 'w1_texture_extra',
  folder: 'cropin_texture', fileFormat: 'CSV'});

Map.setCenter(-51, -27, 6);
Map.addLayer(allPts, {color: 'red'}, 'points');
