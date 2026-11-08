// ============================================================================
// Cropin x MIT – Feature Exploration, Wave 3, Batch 3 (E structure, G terrain/context/geometry, H external)
// Plan: Feature Exploration/00 Workflow Plan.md (v4.1). Written by Claude, 2026-11-06.
//
// HOW TO RUN: paste into code.earthengine.google.com, click Run (Console prints "points __N__"),
// then click RUN on all 7 tasks. Expected 10–60 min each; each task is independent.
//
// FILES PRODUCED (folder cropin_texture):
//   w3_eth.csv       E (H12, H13): ETH 2020 canopy height (10 m) + its SD, at point and mean/SD within 15 m and 50 m
//   w3_gedi.csv      E (H12, H14): GEDI lidar rh98, canopy cover, foliage height diversity, PAI within 100 m
//                    (quality-filtered shots, 3 years ending at each point's cutoff)
//   w3_context.csv   G (H31, H32, H35, H36, H38): WorldCover class shares within 250/500 m, distance to water,
//                    height above drainage, distance to protected areas, WorldClim temperature/rainfall
//   w3_edge.csv      G (H37): Sentinel-2 summer NDVI at point vs a 30–100 m ring around it
//   w3_shape.csv     G (H29, H30): tree-patch edge fraction (WorldCover 2021, patches up to 1,024 px)
//   w3_gft.csv       H: JRC global forest types 2020 V1 (1 naturally regenerating, 10 primary, 20 planted)
//   w3_hansen_tc.csv H: Hansen tree cover 2000 (re-export; lost in batch 1)
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

// ---------- 1. ETH canopy height 2020 ----------
var eth = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').select([0], ['h']);
var ethSd = ee.Image('users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1').select([0], ['h_sd']);
var meanSd = ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true);
var ethImg = eth.rename('eth_h').addBands(ethSd.rename('eth_h_modelsd'))
  .addBands(eth.reduceNeighborhood({reducer: meanSd, kernel: ee.Kernel.circle(15, 'meters')}).rename(['eth_h_mean15', 'eth_h_sd15']))
  .addBands(eth.reduceNeighborhood({reducer: meanSd, kernel: ee.Kernel.circle(50, 'meters')}).rename(['eth_h_mean50', 'eth_h_sd50']));
Export.table.toDrive({collection: sample(ethImg, allPts, 10), description: 'w3_eth', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 2. GEDI L2A/L2B monthly rasters, quality-filtered, within 100 m ----------
function gedi(pts, y) {
  var s = ee.Date.fromYMD(y - 2, 1, 1), e = ee.Date.fromYMD(y + 1, 1, 1);
  var a = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').filterBounds(pts.geometry().bounds()).filterDate(s, e)
    .map(function(im) { return im.updateMask(im.select('quality_flag').eq(1).and(im.select('degrade_flag').eq(0))).select('rh98'); });
  var b = ee.ImageCollection('LARSE/GEDI/GEDI02_B_002_MONTHLY').filterBounds(pts.geometry().bounds()).filterDate(s, e)
    .map(function(im) { return im.updateMask(im.select('l2b_quality_flag').eq(1).and(im.select('degrade_flag').eq(0))).select(['cover', 'fhd_normal', 'pai']); });
  var img = a.median().addBands(b.median());
  var red = ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true).combine(ee.Reducer.count(), '', true);
  return strip(img.reduceRegions({collection: pts.map(function(f) { return f.buffer(100); }), reducer: red, scale: 25, tileScale: 8}));
}
Export.table.toDrive({collection: byCutoff('mb', gedi).merge(byCutoff('qc', gedi)), description: 'w3_gedi', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 3. Context: land-cover mix, water, drainage, protected areas, climate ----------
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');   // 2021
var CL = {tree: 10, shrub: 20, grass: 30, crop: 40, built: 50, water: 80};
var ctx = ee.Image([]);
Object.keys(CL).forEach(function(k) {
  var m = wc.eq(CL[k]);
  ctx = ctx.addBands(m.reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: ee.Kernel.circle(250, 'meters')}).rename('wc_' + k + '_250m'))
           .addBands(m.reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: ee.Kernel.circle(500, 'meters')}).rename('wc_' + k + '_500m'));
});
var water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0).gte(50);
ctx = ctx.addBands(water.fastDistanceTransform(256).sqrt().multiply(30).rename('dist_water_m'))
  .addBands(ee.Image('MERIT/Hydro/v1_0_1').select('hnd').rename('merit_hnd'))
  .addBands(ee.FeatureCollection('WCMC/WDPA/current/polygons').filterBounds(region.buffer(50000))
            .distance({searchRadius: 50000, maxError: 50}).unmask(50000).rename('dist_protected_m'))
  .addBands(ee.Image('WORLDCLIM/V1/BIO').select(['bio01', 'bio12'], ['clim_temp_x10', 'clim_rain_mm']));
Export.table.toDrive({collection: sample(ctx, allPts, 30, 8), description: 'w3_context', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 4. Edge contrast: summer NDVI at point vs a 30–100 m ring ----------
var CS = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var w = [];
for (var i = -10; i <= 10; i++) { var row = []; for (var j = -10; j <= 10; j++) { var r = Math.sqrt(i * i + j * j); row.push(r > 3 && r <= 10 ? 1 : 0); } w.push(row); }
var ring = ee.Kernel.fixed(21, 21, w, -10, -10, true);
function edge(pts, y) {
  var c = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(pts.geometry().bounds())
    .filterDate(y + '-12-01', (y + 1) + '-04-01').linkCollection(CS, ['cs'])
    .map(function(im) { return im.updateMask(im.select('cs').gte(0.6)); }).median();
  var ndvi = c.normalizedDifference(['B8', 'B4']).rename('ndvi_pt');
  var swir = c.select('B11').divide(10000).rename('b11_pt');
  var img = ndvi.addBands(ndvi.convolve(ring).rename('ndvi_ring')).addBands(swir).addBands(swir.convolve(ring).rename('b11_ring'));
  return sample(img, pts, 10);
}
var SEASONS = __SEASONS__;
var edgeAll = ee.FeatureCollection(SEASONS.map(function(y) { return edge(allPts.filter(ee.Filter.eq('season_start', y)), y); })).flatten();
Export.table.toDrive({collection: edgeAll, description: 'w3_edge', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 5. Tree-patch shape (H29) ----------
var tree = wc.eq(10);
var comps = tree.selfMask().connectedComponents({connectedness: ee.Kernel.plus(1), maxSize: 1024});
var edgePx = tree.and(tree.focalMin({radius: 1, kernelType: 'plus', units: 'pixels'}).not());
var shape = edgePx.rename('edge').addBands(comps.select('labels'))
  .reduceConnectedComponents({reducer: ee.Reducer.mean(), labelBand: 'labels', maxSize: 1024}).rename('patch_edge_frac');
Export.table.toDrive({collection: sample(shape.unmask(-1), allPts, 10, 8), description: 'w3_shape', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 6. JRC global forest types 2020 V1 ----------
Export.table.toDrive({collection: sample(ee.Image('JRC/GFC2020_subtypes/V1').select('Map').rename('gft2020').unmask(0), allPts, 10),
  description: 'w3_gft', folder: 'cropin_texture', fileFormat: 'CSV'});
// ---------- 7. Hansen tree cover 2000 ----------
Export.table.toDrive({collection: sample(ee.Image('UMD/hansen/global_forest_change_2025_v1_13').select('treecover2000'), allPts, 30),
  description: 'w3_hansen_tc', folder: 'cropin_texture', fileFormat: 'CSV'});
