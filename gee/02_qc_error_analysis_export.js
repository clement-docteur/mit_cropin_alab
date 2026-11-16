// #####################################################################
// 02_qc_error_analysis_export.js  (Meera, Oct 2026)
// WHAT IT DOES
//   Builds one feature row per Cropin QC polygon (all 433) and exports two CSVs. These are the
//   inputs to notebooks/01_qc_error_analysis.ipynb.
// ASSETS NEEDED (GEE project cropin-510901)
//   projects/cropin-510901/assets/cropin_qc  (433 polygons, id + qc_label)
//   Public: MapBiomas C9 30 m (1985–2023), ETH canopy height 2020 10 m, Google Natural
//   Forests 2020, Sentinel-2 SR harmonized (SCL cloud mask).
// HOW TO RUN
//   Paste into a new script in the Code Editor and press Run. In the Tasks tab, click RUN on both
//   tasks. Files land in Google Drive/cropin_exports. Copy them to data/gee_exports/.
// OUTPUTS
//   qc_error_analysis_A.csv       area, centroid, ETH height mean (inside polygon) and SD (30 m
//                                 circle around the centroid), Natural Forests prob, MapBiomas
//                                 ring fractions, MapBiomas C9 mode class 1985–2023
//   qc_error_analysis_B_ndvi.csv  yearly Sentinel-2 NDVI median, p10 and clear-view count, 2017–2025
//   Every column is defined in data/README.md.
// #####################################################################

// =====================================================================
// Cropin – ERROR ANALYSIS EXPORT
// One row per Cropin QC polygon (all 433), saved as CSVs to your Google Drive.
// Paste into a NEW script in the Earth Engine Code Editor and press Run.
// Then: Tasks tab (top right) → click RUN next to each of the 2 tasks → RUN again in the pop-up.
// When they finish (a few minutes to ~1 hour), the CSVs are in your Google Drive
// in a folder called "cropin_exports". Download both and send them to Claude.
// =====================================================================

// ---- 0. Settings: change this if your assets live in a different project ----
var PROJECT = 'cropin-510901';
var qc = ee.FeatureCollection('projects/' + PROJECT + '/assets/cropin_qc');   // 433 polygons, qc_label

// ---- 1. Data layers ----
// MapBiomas Collection 9 (30m), one band per year 1985–2023. 3 = forest, 9 = plantation.
var mb = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1')
  .regexpRename('classification_', 'mb_');
// Canopy height, ETH 2020 (10m, metres)
var height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').rename('h');
// Google Natural Forests of the World 2020: probability of natural forest, rescaled 0–1
var nf = ee.ImageCollection('projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection')
  .mosaic().select('B0').divide(250).rename('nf_prob');

// ---- 2. Basic info per polygon: id, size, location ----
var base = qc.map(function (f) {
  var c = f.geometry().centroid(1).coordinates();
  return f.set({area_m2: f.geometry().area(1), lon: c.get(0), lat: c.get(1)});
});

// =====================================================================
// TASK A – what the maps said + height + surroundings (fast)
// =====================================================================
// A1. MapBiomas class at the polygon, every year 1985–2023 (most common class inside it)
var a1 = mb.reduceRegions({collection: base, reducer: ee.Reducer.mode(), scale: 30, tileScale: 4});

// A2. Average canopy height and Natural Forests probability inside the polygon
var a2 = ee.Image.cat(height.rename('height_mean'), nf)
  .reduceRegions({collection: a1, reducer: ee.Reducer.mean(), scale: 10, tileScale: 4});

// A3. Height variability (30 m circle) and what surrounds the patch (a ring 30–150 m out)
var lc2020 = mb.select('mb_2020');
var ringLayers = lc2020.eq(3).rename('ring_forest_frac')
  .addBands(lc2020.eq(9).rename('ring_plant_frac'));
var a3 = a2.map(function (f) {
  var g = f.geometry();
  var circle = g.centroid(1).buffer(30);
  var ring = g.buffer(150, 1).difference(g.buffer(30, 1), 1);
  var hsd = height.reduceRegion({reducer: ee.Reducer.stdDev(), geometry: circle, scale: 10}).get('h');
  var surround = ringLayers.reduceRegion({reducer: ee.Reducer.mean(), geometry: ring, scale: 30});
  return f.set({
    height_sd_30m: hsd,
    ring_forest_frac: surround.get('ring_forest_frac'),
    ring_plant_frac: surround.get('ring_plant_frac')
  });
});

var mbCols = [];
for (var y = 1985; y <= 2023; y++) { mbCols.push('mb_' + y); }
Export.table.toDrive({
  collection: a3,
  description: 'qc_error_analysis_A',
  folder: 'cropin_exports',
  fileFormat: 'CSV',
  selectors: ['id', 'qc_label', 'area_m2', 'lon', 'lat', 'height_mean', 'height_sd_30m',
              'nf_prob', 'ring_forest_frac', 'ring_plant_frac'].concat(mbCols)
});

// =====================================================================
// TASK B – greenness (NDVI) history from Sentinel-2, per year 2017–2025 (slower)
// =====================================================================
var box = qc.geometry().bounds(1);
var years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
function yearly(y) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(box)
    .filterDate(y + '-01-01', (y + 1) + '-01-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .map(function (img) {
      var scl = img.select('SCL');                        // keep vegetation (4) and bare soil (5)
      return img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        .updateMask(scl.eq(4).or(scl.eq(5)));
    });
  return ee.Image.cat(
    col.median().rename('ndvi_med_' + y),                               // typical greenness
    col.reduce(ee.Reducer.percentile([10])).rename('ndvi_p10_' + y),    // low point (harvest dips)
    col.count().rename('n_obs_' + y));                                  // how many clear views
}
var ndviStack = ee.Image.cat(years.map(yearly));
var b = ndviStack.reduceRegions({collection: base, reducer: ee.Reducer.mean(), scale: 10, tileScale: 16});

var ndviCols = [];
years.forEach(function (y) { ndviCols.push('ndvi_med_' + y, 'ndvi_p10_' + y, 'n_obs_' + y); });
Export.table.toDrive({
  collection: b,
  description: 'qc_error_analysis_B_ndvi',
  folder: 'cropin_exports',
  fileFormat: 'CSV',
  selectors: ['id', 'qc_label'].concat(ndviCols)
});

print('Two export tasks are ready. Open the Tasks tab (top right) and click RUN on each.');
print('Number of polygons:', qc.size());
