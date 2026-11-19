// Cropin's forest layer and SAG observations at the harvest-cycle points
// ---------------------------------------------------------------------
// Feeds section 8 of harvest_cycles/harvest_cycles_findings.ipynb with Cropin's exact layers instead of stand-ins.
// The logic is copied from "Google Earth Engine Base/Forest layer" and "Google Earth Engine Base/Multiyear SAG
// Plantation Code" (same datasets, thresholds, weights and masks).
//
// How to run:
//   1. Upload public_data/harvest_cycles/points.csv as a table asset (Assets > New > CSV file) and set POINTS below.
//      The file holds locations only; Cropin's labels are not in it.
//   2. Run the script, then start both exports in the Tasks tab.
//   3. Save the two CSV files in public_data/harvest_cycles/ under the names given in `description`.
//      The notebook then uses them automatically and applies Cropin's "Final Forest Mask" rule to the exported parts:
//      forest_score >= 0.75, not a MapBiomas 10 m plantation, MapBiomas 10 m beta forest, canopy >= 4 m.

var POINTS = 'projects/YOUR_PROJECT/assets/harvest_cycles_points';   // <-- edit
var TARGET_YEAR = 2020;                                                // year of Cropin's forest layer
var FIRST_SEASON = 2019, LAST_SEASON = 2025;   // SAG seasons: 1 Oct of year i up to (not including) 31 Jan of year i+1, as in Cropin's script

var points = ee.FeatureCollection(POINTS).map(function (f) {
  return ee.Feature(ee.Geometry.Point([ee.Number(f.get('lon')), ee.Number(f.get('lat'))]), {pid: f.get('pid')});
});

// ======================================================================================
// 1. Cropin's forest layer (from "Forest layer")
// ======================================================================================
var gfc = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');
var lossyear = gfc.select('lossyear').unmask(-99);
var forest_loss = lossyear.gt(0).and(lossyear.lte(20));
var hansen_forest = gfc.select('treecover2000').gte(30).and(forest_loss.not());

var mb10 = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1')
  .select('classification_' + TARGET_YEAR);
var mb10_beta = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection_S2_beta/collection_LULC_S2_beta')
  .select('classification_' + TARGET_YEAR);
var mb30 = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_coverage_v2')
  .select('classification_' + TARGET_YEAR);
var mb_forest_30m = mb30.eq(3);
var mb_plantation_10m = mb10_beta.eq(9).or(mb10.eq(9));
var mb_forest_10m_beta = mb10_beta.eq(3);

var jrc_forest = ee.ImageCollection('JRC/GFC2020/V2').mosaic().gt(0.5).selfMask();
var tmf = ee.ImageCollection('projects/JRC/TMF/v1_2024/AnnualChanges').mosaic().select('Dec' + TARGET_YEAR);
var tmf_forest = tmf.lte(2).or(tmf.eq(4)).selfMask();
var canopy = ee.ImageCollection('projects/sat-io/open-datasets/facebook/meta-canopy-height').mosaic();

// Weighted ensemble with Cropin's weights: Hansen, JRC, MapBiomas 30 m, TMF = 2, 2, 3, 3 in 2020, else 2, 0, 3, 3
var w = TARGET_YEAR === 2020 ? [2, 2, 3, 3] : [2, 0, 3, 3];
var forest_score = hansen_forest.unmask(0).multiply(w[0])
  .add(jrc_forest.unmask(0).multiply(w[1]))
  .add(mb_forest_30m.unmask(0).multiply(w[2]))
  .add(tmf_forest.unmask(0).multiply(w[3]))
  .divide(w[0] + w[1] + w[2] + w[3]);

// Every part is read at the pixel under the point (scale 1 m, so no 10 m grid can shift the read), and the canopy
// is also averaged over a 10 m x 10 m square centred on the point. The notebook applies the "Final Forest Mask" rule.
var layer_bands = ee.Image.cat([
  forest_score.rename('forest_score'),
  hansen_forest.unmask(0).rename('hansen_forest'),
  jrc_forest.unmask(0).rename('jrc_forest'),
  mb_forest_30m.unmask(0).rename('mb_forest_30m'),
  tmf_forest.unmask(0).rename('tmf_forest'),
  mb_plantation_10m.unmask(0).rename('mb_plantation_10m'),
  mb_forest_10m_beta.unmask(0).rename('mb_forest_10m_beta'),
  canopy.rename('canopy_at_point')
]);

var layer_at_points = layer_bands.reduceRegions({collection: points, reducer: ee.Reducer.first(), scale: 1});
var canopy_10m = canopy.rename('canopy_10m').reduceRegions({
    collection: points.map(function (f) { return f.buffer(5).bounds(); }),
    reducer: ee.Reducer.mean().setOutputs(['canopy_10m']),
    scale: 1});
var layer_table = ee.Join.inner().apply(layer_at_points, canopy_10m, ee.Filter.equals({leftField: 'pid', rightField: 'pid'}))
  .map(function (pair) {
    var a = ee.Feature(pair.get('primary')), b = ee.Feature(pair.get('secondary'));
    return ee.Feature(null, a.toDictionary()).set('canopy_10m', b.get('canopy_10m'));
  });

Export.table.toDrive({
  collection: layer_table,
  description: 'cropin_forest_layer',
  fileFormat: 'CSV',
  selectors: ['pid', 'forest_score', 'hansen_forest', 'jrc_forest', 'mb_forest_30m', 'tmf_forest',
              'mb_plantation_10m', 'mb_forest_10m_beta', 'canopy_at_point', 'canopy_10m']
});

// ======================================================================================
// 2. SAG observations (from "Multiyear SAG Plantation Code")
//    Every valid SAG value at every point, Oct-Jan of each season. The notebook applies Cropin's Bayesian update
//    (plantation 160 +- 9, other 150 +- 9, start 0.5, clamp 0.1-0.9, mean over the seasons).
// ======================================================================================
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var csPlusBands = csPlus.first().bandNames();

function maskLowQA(image) {
  return image.updateMask(image.select('cs').gte(0.6));
}

function addGreenAngle(s2_dataset) {          // copied unchanged
  var yB = s2_dataset.select('B2').divide(10000);
  var yG = s2_dataset.select('B3').divide(10000);
  var yR = s2_dataset.select('B4').divide(10000);
  var xB = 0.4924, xG = 0.5598, xR = 0.6646;
  var dx1 = xB - xG, dy1 = yB.subtract(yG);
  var dx2 = xR - xG, dy2 = yR.subtract(yG);
  var dotProduct = dy1.multiply(dy2).add(dx1 * dx2);
  var mag1 = dy1.pow(2).add(Math.pow(dx1, 2)).sqrt();
  var mag2 = dy2.pow(2).add(Math.pow(dx2, 2)).sqrt();
  var cosTheta = dotProduct.divide(mag1.multiply(mag2)).clamp(-1.0, 1.0);
  var angleDeg = cosTheta.acos().multiply(180 / Math.PI).rename('SAG');
  var band_value_mask = yG.lte(0.05).and(yR.lte(0.05)).and(yB.lte(0.05));
  return s2_dataset.addBands(angleDeg).mask(band_value_mask);
}

var observations = ee.FeatureCollection([]);
for (var i = FIRST_SEASON; i <= LAST_SEASON; i++) {
  var season = i;
  var sag = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(season + '-10-01', (season + 1) + '-01-31')
    .filterBounds(points)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
    .linkCollection(csPlus, csPlusBands)
    .map(maskLowQA)
    .map(addGreenAngle)
    .select('SAG');
  var season_obs = sag.map(function (img) {
    return img.reduceRegions({
        collection: points.filterBounds(img.geometry()),
        reducer: ee.Reducer.first().setOutputs(['SAG']),
        scale: 10})
      .filter(ee.Filter.notNull(['SAG']))
      .map(function (f) {
        return f.set({season: season, time: img.get('system:time_start'), image: img.get('system:index')})
                .setGeometry(null);
      });
  }).flatten();
  observations = observations.merge(season_obs);
}

Export.table.toDrive({
  collection: observations,
  description: 'cropin_sag_observations',
  fileFormat: 'CSV',
  selectors: ['pid', 'season', 'time', 'image', 'SAG']
});
