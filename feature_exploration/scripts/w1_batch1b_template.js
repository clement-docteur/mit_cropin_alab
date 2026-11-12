// ============================================================================
// Cropin x MIT – Feature Exploration, Wave 1, Batch 1b (re-run of the failed w1_texture_extra)
// Written by Claude, 2026-11-05. The original task failed with "User memory limit exceeded"
// after 513,844 EECU-seconds. Fixes: (1) split into 4 smaller tasks; (2) SNIC now runs only
// within 300 m of each point; (3) segment size cap 1,024 -> 256 px; (4) tileScale 16.
//
// HOW TO RUN: paste into code.earthengine.google.com, click Run (Console prints
// "MapBiomas points __N_MB__" and "QC points __N_QC__"), then click RUN on all 4 tasks.
//
// FILES PRODUCED (folder cropin_texture):
//   w1_tex_glcm_dir.csv     GLCM on B4/B12 + directional B8 contrast (MapBiomas 2021-22, QC 4 seasons)
//   w1_snic_mb.csv          SNIC segment stats, MapBiomas points, 2021-22
//   w1_snic_qc_2025.csv     SNIC segment stats, QC points, 2025-26 (395 points use this season)
//   w1_snic_qc_2022_24.csv  SNIC segment stats, QC points, 2022-23 to 2024-25
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

var CS = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
function s2summer(y, reg) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(reg)
    .filterDate(y + '-12-01', (y + 1) + '-04-01').linkCollection(CS, ['cs'])
    .map(function(i) { return i.updateMask(i.select('cs').gte(0.6)); })
    .select(['B4', 'B8', 'B11', 'B12']).median();
}
function extras(img, part, pts) {
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var x = img.divide(10000).addBands(ndvi);
  // SNIC segments (seed spacing 10 px = 100 m)
  if (part === 'snic') {
  // Clip to 300 m around the points so SNIC only runs where we sample (the 5 Nov run without this
  // exceeded GEE's memory limit after 513,844 EECU-seconds).
  var xs = x.clipToCollection(pts.map(function(f) { return f.buffer(300); }));
  var snic = ee.Algorithms.Image.Segmentation.SNIC({image: xs.select(['B4', 'B8', 'B11']),
    size: 10, compactness: 1, connectivity: 8});
  var cl = snic.select('clusters');
  var segSd = xs.select(['B8', 'B11', 'NDVI']).addBands(cl)
    .reduceConnectedComponents({reducer: ee.Reducer.stdDev(), labelBand: 'clusters', maxSize: 256})
    .rename(['seg_sd_B8', 'seg_sd_B11', 'seg_sd_NDVI']);
  var segN = cl.connectedPixelCount({maxSize: 256, eightConnected: true}).rename('seg_px');
  return segSd.addBands(segN);
  }
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
  return dc.addBands(tex);
}

function run(pts, y, part) {
  return extras(s2summer(y, pts.geometry().bounds()), part, pts)
    .sampleRegions({collection: pts, scale: 10, geometries: false, tileScale: 16})
    .map(function(f) { return f.set('season_start', y); });
}
var cheap = run(mbPts, 2021, 'cheap').merge(ee.FeatureCollection([2022, 2023, 2024, 2025].map(function(y) {
  return run(qcPts, y, 'cheap'); })).flatten());
Export.table.toDrive({collection: cheap, description: 'w1_tex_glcm_dir', folder: 'cropin_texture', fileFormat: 'CSV'});
Export.table.toDrive({collection: run(mbPts, 2021, 'snic'), description: 'w1_snic_mb', folder: 'cropin_texture', fileFormat: 'CSV'});
Export.table.toDrive({collection: run(qcPts, 2025, 'snic'), description: 'w1_snic_qc_2025', folder: 'cropin_texture', fileFormat: 'CSV'});
Export.table.toDrive({collection: ee.FeatureCollection([2022, 2023, 2024].map(function(y) {
  return run(qcPts, y, 'snic'); })).flatten(), description: 'w1_snic_qc_2022_24', folder: 'cropin_texture', fileFormat: 'CSV'});
