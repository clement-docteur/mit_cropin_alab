// #####################################################################
// REVIEW 2026-10-29 (logic checked against gee/cropin/*.js; no change needed)
//   Forest layer: weights 2/2/3/3, Hansen rule, JRC > 0.5, TMF rule, the 10 m plantation
//   exclusion, beta-forest and Meta >= 4 m filters, and the 0.75 cut all match
//   cropin_forest_layer.js. unmask(0) here is equivalent to Cropin's updateMask chain.
//   SAG formula, dark-pixel rule, CS+ >= 0.6 and CLOUDY_PIXEL_PERCENTAGE < 90 all match.
//   Handle in Python (done in notebook 02):
//    - Overlapping S2 tiles give two rows per (id, date): 10,837 pairs. Cropin's per-pixel
//      iterate also ingests both, so keep them for the faithful posterior and average per date
//      for seasonal means.
//    - Cropin's season is Oct 1 to Jan 30 (filterDate end is exclusive).
//    - Meta columns export as generic mean/stdDev/p10/p90/count; renamed to meta_*.
//   Assets: projects/cropin-510901/assets/cropin_qc. Outputs go to data/gee_exports/.
// #####################################################################
// =====================================================================
// Cropin – apply CROPIN'S OWN layers and SAG to our 433 QC polygons
// Written 2026-10-28. It reuses the logic in gee/cropin/*.js (shared by Cropin on 10/28).
// Paste into a NEW script in the Earth Engine Code Editor (project cropin-510901) and press Run.
// Then open the Tasks tab and click RUN on each of the 2 tasks. The CSVs go to Google Drive
// in the folder "cropin_exports". Put them in data/gee_exports/.
//
// TASK C  qc_cropin_layers.csv   one row per polygon:
//   - Cropin's forest-layer ensemble: score, final forest flag and each component;
//   - MapBiomas 10m (Collection 2) and 10m S2-beta classes for every year (mode inside the
//     polygon), plus MapBiomas Collection 10 (30m), so we can benchmark against the SAME
//     MapBiomas products Cropin uses (and reconcile Julia's 2025 number);
//   - Meta 1m canopy height inside the polygon: mean, std dev, p10/p90 (structure *inside*
//     the stand, the follow-up to finding 3).
// TASK D  qc_sag_timeseries.csv  one row per (polygon, Sentinel-2 image), 2019–2025, all months:
//   - mean SAG of the polygon's valid pixels, and the number of pixels;
//   - the Bayesian multi-year posterior is then rebuilt in Python from this table, so the
//     Oct–Jan (summer) vs Jun–Aug (winter) windows can be compared.
//
// Differences from Cropin's code (deliberate, documented):
//   - Cropin's addGreenAngle() uses image.mask(...), which REPLACES the Cloud Score+ mask.
//     Here we use updateMask(...) so both masks apply. Their dark-pixel rule (B2, B3 and B4
//     reflectance all ≤ 0.05) mostly removes clouds anyway.
//   - SAG is averaged over the polygon per image (a polygon-level time series), whereas Cropin
//     runs the Bayes update per pixel. With polygons of about 3 pixels the two are close, but
//     not identical.
// =====================================================================

var PROJECT = 'cropin-510901';
var qc = ee.FeatureCollection('projects/' + PROJECT + '/assets/cropin_qc');   // 433 polygons, id + qc_label
var TARGET_YEAR = 2020;

// ---------------- Cropin forest layer (same logic as gee/cropin/cropin_forest_layer.js) ----------------
var gfc = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');
var lossRecent = gfc.select('lossyear').unmask(-99).gt(0).and(gfc.select('lossyear').unmask(-99).lte(20));
var hansen_forest = gfc.select('treecover2000').gte(30).and(lossRecent.not());

var mb10    = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1');
var mb10b   = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection_S2_beta/collection_LULC_S2_beta');
var mb30c10 = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_coverage_v2');

var y = 'classification_' + TARGET_YEAR;
var mbi_forest_30m = mb30c10.select(y).eq(3);
var mbi_plant10     = mb10.select(y).eq(9);
var mbi_forest10b   = mb10b.select(y).eq(3);
var mbi_plant10b    = mb10b.select(y).eq(9);
var combined_plant  = mbi_plant10b.or(mbi_plant10);

var jrc_forest = ee.ImageCollection('JRC/GFC2020/V2').mosaic().gt(0.5).selfMask();
var tmfY = ee.ImageCollection('projects/JRC/TMF/v1_2024/AnnualChanges').mosaic().select('Dec' + TARGET_YEAR);
var tmf_forest = tmfY.lte(2).or(tmfY.eq(4)).selfMask();

// weights for 2020: Hansen 2, JRC 2, MapBiomas30 3, TMF 3 (sum 10)
var score = hansen_forest.unmask(0).multiply(2)
  .add(jrc_forest.unmask(0).multiply(2))
  .add(mbi_forest_30m.unmask(0).multiply(3))
  .add(tmf_forest.unmask(0).multiply(3))
  .divide(10);

var meta = ee.ImageCollection('projects/sat-io/open-datasets/facebook/meta-canopy-height').mosaic().rename('meta_h');
var heightOK = meta.gte(4);

// Final Cropin forest = score ≥ 0.75, not MapBiomas-10m plantation, MapBiomas-10m-beta forest, Meta height ≥ 4 m
var cropinForest = score.gte(0.75)
  .and(combined_plant.unmask(0).eq(0))
  .and(mbi_forest10b.unmask(0))
  .and(heightOK.unmask(0));

var layers10 = ee.Image.cat(
  score.rename('cf_score'),
  cropinForest.rename('cf_forest_final'),
  hansen_forest.unmask(0).rename('cf_hansen'),
  jrc_forest.unmask(0).rename('cf_jrc'),
  mbi_forest_30m.unmask(0).rename('cf_mb30c10_forest'),
  tmf_forest.unmask(0).rename('cf_tmf'),
  combined_plant.unmask(0).rename('cf_mb10_plant_any'),
  mbi_forest10b.unmask(0).rename('cf_mb10beta_forest'),
  heightOK.unmask(0).rename('cf_meta_ge4')
);

// ---------------- TASK C: one row per polygon ----------------
var c1 = layers10.reduceRegions({collection: qc, reducer: ee.Reducer.mean(), scale: 10, tileScale: 4});

// All years of the three MapBiomas products (mode = most common class inside the polygon)
var mbAll = ee.Image.cat(
  mb10.regexpRename('classification_', 'mb10c2_'),
  mb10b.regexpRename('classification_', 'mb10beta_'),
  mb30c10.regexpRename('classification_', 'mb30c10_'));
var c2 = mbAll.reduceRegions({collection: c1, reducer: ee.Reducer.mode(), scale: 10, tileScale: 4});

// Meta 1 m canopy height INSIDE the polygon
var c3 = meta.reduceRegions({
  collection: c2,
  reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true)
           .combine(ee.Reducer.percentile([10, 90]), '', true)
           .combine(ee.Reducer.count(), '', true),
  scale: 1, tileScale: 4});

Export.table.toDrive({
  collection: c3.map(function (f) { return f.setGeometry(null); }),
  description: 'qc_cropin_layers',
  folder: 'cropin_exports',
  fileFormat: 'CSV'
});

// ---------------- TASK D: SAG time series per polygon ----------------
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var csBands = csPlus.first().bandNames();

function addSAG(img) {                      // same formula as Cropin's addGreenAngle()
  var yB = img.select('B2').divide(10000), yG = img.select('B3').divide(10000), yR = img.select('B4').divide(10000);
  var xB = 0.4924, xG = 0.5598, xR = 0.6646;  // band centre wavelengths (µm)
  var dx1 = xB - xG, dy1 = yB.subtract(yG);   // green → blue
  var dx2 = xR - xG, dy2 = yR.subtract(yG);   // green → red
  var dot = dy1.multiply(dy2).add(dx1 * dx2);
  var m1 = dy1.pow(2).add(dx1 * dx1).sqrt(), m2 = dy2.pow(2).add(dx2 * dx2).sqrt();
  var sag = dot.divide(m1.multiply(m2)).clamp(-1, 1).acos().multiply(180 / Math.PI).rename('SAG');
  var dark = yG.lte(0.05).and(yR.lte(0.05)).and(yB.lte(0.05));
  return sag.updateMask(dark)               // updateMask (keeps the CS+ mask); Cropin uses .mask()
    .set('system:time_start', img.get('system:time_start'))
    .set('tile', img.get('MGRS_TILE'));
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(qc)
  .filterDate('2019-01-01', '2026-02-01')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
  .linkCollection(csPlus, csBands)
  .map(function (img) { return img.updateMask(img.select('cs').gte(0.6)); })
  .map(addSAG);

var sagTable = s2.map(function (img) {
  var fc = qc.filterBounds(img.geometry());
  var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
  return img.reduceRegions({
      collection: fc,
      reducer: ee.Reducer.mean().combine(ee.Reducer.count(), '', true),
      scale: 10})
    .filter(ee.Filter.gt('count', 0))
    .map(function (f) { return f.set({date: date, tile: img.get('tile')}).setGeometry(null); });
}).flatten();

Export.table.toDrive({
  collection: sagTable,
  description: 'qc_sag_timeseries',
  folder: 'cropin_exports',
  fileFormat: 'CSV',
  selectors: ['id', 'qc_label', 'date', 'tile', 'mean', 'count']   // mean = SAG in degrees, count = valid pixels
});

print('Two export tasks ready (qc_cropin_layers, qc_sag_timeseries). Open Tasks and click RUN on each.');
print('Bands available in MapBiomas 10m C2:', mb10.bandNames());
print('Bands available in MapBiomas 10m S2 beta:', mb10b.bandNames());
