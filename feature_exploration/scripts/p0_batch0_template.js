// ============================================================================
// Cropin x MIT – Feature Exploration, Phase 0, Batch 0
// Plan: Feature Exploration/00 Workflow Plan.md (v4.1). Written by Claude, 2026-11-05.
//
// HOW TO RUN (about 3 minutes of clicking):
//   1. Paste this whole script into code.earthengine.google.com and click Run.
//      The Console should print: "MapBiomas points __N_MB__" and "QC points __N_QC__".
//   2. Open the Tasks tab (top right). Six tasks appear. Click RUN on each,
//      keep the default settings, and click RUN again in the pop-up.
//   3. That's it. Files land in Drive folder "cropin_texture" and sync to the Mac.
//      Expected time: roughly 5–30 min per task (they run in parallel).
//
// FILES PRODUCED (folder cropin_texture):
//   p0_mb_s2.csv          MapBiomas points: S2 bands (2 windows), texture, clear-image counts
//   p0_qc_s2_seasons.csv  QC points: same features for 4 seasons (2022-23 ... 2025-26)
//   p0_qc_ndvi_monthly.csv QC points: monthly NDVI/NBR Jan 2022 – Oct 2026 (to date clearings)
//   p0_nfw.csv            Natural Forests of the World 2020 probability, all points
//   p0_mapbiomas10m.csv   MapBiomas 10 m map classes (all years), all points
//   p0_chm.csv            Meta canopy height (mean, sd, p90 in 5 m and 15 m radius), all points
// Each task is separate, so if one fails the others still finish. Tell Claude which failed.
// ============================================================================

// ---------- points: [lon, lat, label (1 = plantation, 0 = forest), id] ----------
var MB_RAW = __MB_RAW__;
var QC_RAW = __QC_RAW__;
function toFC(raw, set) {
  return ee.FeatureCollection(raw.map(function(r) {
    return ee.Feature(ee.Geometry.Point([r[0], r[1]]), {label: r[2], id: r[3], set: set});
  }));
}
var mbPts = toFC(MB_RAW, 'mb');
var qcPts = toFC(QC_RAW, 'qc');
var allPts = mbPts.merge(qcPts);
print('MapBiomas points', mbPts.size(), 'QC points', qcPts.size());

// ---------- Sentinel-2 with Cloud Score+ (cs >= 0.6 = clear) ----------
var CS = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
function s2(start, end, region, masked) {
  var c = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region).filterDate(start, end).linkCollection(CS, ['cs']);
  if (masked) c = c.map(function(img) { return img.updateMask(img.select('cs').gte(0.6)); });
  return c.select(BANDS);
}
// Median composite + clear-image count (n_clear) + all-image count (n_all)
function composite(start, end, region, prefix) {
  var clear = s2(start, end, region, true);
  var all = s2(start, end, region, false);
  var med = clear.median();
  var ndvi = med.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var out = med.addBands(ndvi);
  out = out.rename(out.bandNames().map(function(b) { return ee.String(prefix).cat(b); }));
  return out.addBands(clear.select('B8').count().rename(prefix + 'n_clear'))
            .addBands(all.select('B8').count().rename(prefix + 'n_all'));
}

// ---------- texture (same settings as the 4 Nov test, for comparability) ----------
function toGrey(img) {   // GLCM needs integers: rescale to 0–255 over fixed ranges
  return ee.Image.cat([
    img.select('su_B8').unitScale(0, 6000).multiply(255).clamp(0, 255).toByte().rename('B8'),
    img.select('su_B5').unitScale(0, 3000).multiply(255).clamp(0, 255).toByte().rename('B5'),
    img.select('su_B11').unitScale(0, 4000).multiply(255).clamp(0, 255).toByte().rename('B11'),
    img.select('su_NDVI').unitScale(0.2, 1.0).multiply(255).clamp(0, 255).toByte().rename('NDVI')
  ]);
}
var KEEP = ['contrast', 'ent', 'idm', 'asm', 'corr', 'var'];
function texture(img) {
  var grey = toGrey(img);
  var out = ee.Image([]);
  [1, 2, 3].forEach(function(r) {
    var w = 'w' + (2 * r + 1);
    var g = grey.glcmTexture({size: r});
    ['B8', 'B5', 'B11', 'NDVI'].forEach(function(b) {
      KEEP.forEach(function(m) {
        out = out.addBands(g.select(b + '_' + m).rename(b + '_' + m + '_' + w));
      });
    });
    out = out.addBands(img.select(['su_B8', 'su_NDVI'])
      .reduceNeighborhood({reducer: ee.Reducer.stdDev(), kernel: ee.Kernel.square(r)})
      .rename(['B8_sd_' + w, 'NDVI_sd_' + w]));
  });
  return out;
}
// One season = summer window Dec–Mar ("su_", used for texture) + SAG window Oct–Jan ("sg_")
function season(y, region) {
  var su = composite(y + '-12-01', (y + 1) + '-04-01', region, 'su_');
  var sg = composite(y + '-10-01', (y + 1) + '-02-01', region, 'sg_');
  return su.addBands(texture(su)).addBands(sg);
}

// ---------- 1. MapBiomas points: season 2021-22 (inside the 2013–2022 label window) ----------
var mbImg = season(2021, mbPts.geometry().bounds());
Export.table.toDrive({
  collection: mbImg.sampleRegions({collection: mbPts, scale: 10, geometries: false, tileScale: 4}),
  description: 'p0_mb_s2', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 2. QC points: four seasons; Claude picks the last one before each clearing ----------
var qcSeasons = ee.FeatureCollection([2022, 2023, 2024, 2025].map(function(y) {
  return season(y, qcPts.geometry().bounds())
    .sampleRegions({collection: qcPts, scale: 10, geometries: false, tileScale: 4})
    .map(function(f) { return f.set('season_start', y); });
})).flatten();
Export.table.toDrive({collection: qcSeasons, description: 'p0_qc_s2_seasons',
  folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 3. QC monthly NDVI / NBR (to estimate each clearing month) ----------
var months = ee.List.sequence(0, 57);   // Jan 2022 … Oct 2026
var qcMonthly = ee.FeatureCollection(months.map(function(i) {
  var start = ee.Date('2022-01-01').advance(i, 'month');
  var c = s2(start, start.advance(1, 'month'), qcPts.geometry().bounds(), true)
    .map(function(img) {
      return img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        .addBands(img.normalizedDifference(['B8', 'B12']).rename('NBR'));
    });
  var empty = ee.Image.constant([0, 0, 0]).rename(['NDVI', 'NBR', 'n_clear']).updateMask(0);
  var img = ee.Image(ee.Algorithms.If(c.size().gt(0),
    c.median().addBands(c.select('NDVI').count().rename('n_clear')).toFloat(),
    empty.toFloat()));
  return img.reduceRegions({collection: qcPts, reducer: ee.Reducer.first(), scale: 10, tileScale: 4})
    .map(function(f) { return f.set('month', start.format('YYYY-MM')); });
})).flatten();
Export.table.toDrive({collection: qcMonthly.select(['id', 'label', 'month', 'NDVI', 'NBR', 'n_clear'], null, false),
  description: 'p0_qc_ndvi_monthly', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 4. Natural Forests of the World 2020 (probability band B0) ----------
var nfw = ee.ImageCollection('projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection')
  .mosaic().select('B0').rename('nfw');
Export.table.toDrive({
  collection: nfw.sampleRegions({collection: allPts, scale: 10, geometries: false, tileScale: 4}),
  description: 'p0_nfw', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 5. MapBiomas 10 m map, all years (competitor / baseline) ----------
var mb10 = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1');
Export.table.toDrive({
  collection: mb10.sampleRegions({collection: allPts, scale: 10, geometries: false, tileScale: 4}),
  description: 'p0_mapbiomas10m', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 6. Meta canopy height (1 m): mean, sd, p90 within 5 m and 15 m of each point ----------
var chm = ee.ImageCollection('projects/meta-forest-monitoring-okw37/assets/CanopyHeight')
  .mosaic().select([0], ['h']);
var red = ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true)
  .combine(ee.Reducer.percentile([90]), '', true);
function chmAt(radius) {
  return chm.reduceRegions({collection: allPts.map(function(f) { return f.buffer(radius); }),
    reducer: red, scale: 1, tileScale: 4})
    .map(function(f) { return ee.Feature(null, f.toDictionary()).set('radius_m', radius); });
}
Export.table.toDrive({collection: chmAt(5).merge(chmAt(15)), description: 'p0_chm',
  folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- quick look ----------
Map.setCenter(-51, -27, 6);
Map.addLayer(mbPts, {color: 'blue'}, 'MapBiomas points');
Map.addLayer(qcPts, {color: 'red'}, 'QC points');
