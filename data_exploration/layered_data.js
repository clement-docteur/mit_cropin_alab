// =============================================================================
// Validation points (true labels) vs. MapBiomas 10m (predicted)
// Classes simplified to: forest / plantation / other
//
// HOW TO RUN (two phases, avoids timeouts):
//   1) Leave EXPORTED = false, run, then go to the Tasks tab and click RUN on
//      the export task. Wait for it to finish (a few minutes).
//   2) Set EXPORTED = true and run again. Everything now loads from the
//      finished result, so it's fast and won't time out.
// =============================================================================

// ---- SETTINGS ---------------------------------------------------------------
var year     = 2022;   // points labeled 1985-2022, 10m map starts 2017 -> 2017-2022
var EXPORTED = false;  // flip to true after the export task finishes
var points   = ee.FeatureCollection('projects/cropin-510901/assets/mapbiomas_85k_points_validation');
var RESULT   = 'projects/cropin-510901/assets/points_vs_mapbiomas_' + year;
// -----------------------------------------------------------------------------

// Import MapBiomas predicted data
var Palette  = require('users/mapbiomas-global/LULC:LULC_palette.js');
var vis_LULC = Palette.get('vis_LULC');
var lulc = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1')
  .select('classification_' + year);

// Simplify any MapBiomas code: 1 = forest, 2 = plantation (silviculture), 0 = other
var forestCodes = [3, 6, 4, 7, 5, 49];
var pred = lulc.remap(forestCodes.concat([9]), [1, 1, 1, 1, 1, 1, 2], 0).rename('pred');
var names = ee.Dictionary({'0': 'other', '1': 'forest', '2': 'plantation'});

// ---- Base layers ------------------------------------------------------------
Map.addLayer(lulc, vis_LULC, 'MapBiomas full legend', false, 0.5);
Map.addLayer(pred.selfMask(), {min: 1, max: 2, palette: ['1f8d49', 'ff7f00']},
             'Predicted: forest (green) / plantation (orange)', true, 0.6);
Map.setCenter(-51.0, -15.0, 4);

if (!EXPORTED) {
  // ---- PHASE 1: compute the comparison once and save it as an asset ---------
  var truthCol = 'CLASS_' + year;
  var cmp = pred.sampleRegions({
    collection: points.filter(ee.Filter.eq('BORDA_' + year, 0))  // skip edge points
                      .select([truthCol]),                       // keep only what we need
    scale: 10, geometries: true, tileScale: 16
  }).map(function (f) {
    var t = ee.Number(f.get(truthCol));
    var trueCode = ee.Number(ee.Algorithms.If(t.eq(9), 2,
                   ee.Algorithms.If(ee.List(forestCodes).contains(t), 1, 0)));
    var predCode = ee.Number(f.get('pred'));
    // Only readable fields, so the Inspector shows e.g. TRUE: forest, PREDICTED: plantation
    return ee.Feature(f.geometry(), {
      TRUE: names.get(trueCode.format()),
      PREDICTED: names.get(predCode.format()),
      true_code: trueCode, pred_code: predCode,
      agree: trueCode.eq(predCode)
    });
  });

  Export.table.toAsset({
    collection: cmp,
    description: 'points_vs_mapbiomas_' + year,
    assetId: RESULT
  });
  print('Phase 1: open the Tasks tab and click RUN. Then set EXPORTED = true.');

} else {
  // ---- PHASE 2: load the saved result (fast) --------------------------------
  var res = ee.FeatureCollection(RESULT);

  Map.addLayer(res.filter(ee.Filter.eq('agree', 1)).style({color: '00e5ff', pointSize: 4}),
               {}, 'Points: AGREE (cyan)', false);
  Map.addLayer(res.filter(ee.Filter.eq('agree', 0)).style({color: 'ff00ff', pointSize: 5}),
               {}, 'Points: DISAGREE (magenta)');

  // Rows = true, columns = predicted, in the order other / forest / plantation
  var cm = res.errorMatrix('true_code', 'pred_code', [0, 1, 2]);
  print('Confusion matrix (rows TRUE, cols PREDICTED: other / forest / plantation)');
  print(cm);
  print('Overall accuracy');
  print(cm.accuracy());
  print('Per-class accuracy, true-label side (other / forest / plantation)');
  print(cm.producersAccuracy());

  // ---- Click popup: nearest point's TRUE vs PREDICTED, in words -------------
  var readout = ui.Label('Zoom in and click a point', {position: 'top-center', whiteSpace: 'pre'});
  Map.add(readout);
  Map.style().set('cursor', 'crosshair');

  Map.onClick(function (coords) {
    readout.setValue('Loading...');
    var pt = ee.Geometry.Point(coords.lon, coords.lat);
    var hit = res.filterBounds(pt.buffer(50)).first();
    ee.Dictionary({
      t: ee.Algorithms.If(hit, ee.Feature(hit).get('TRUE'), null),
      p: names.get(ee.Number(pred.reduceRegion(ee.Reducer.first(), pt, 10).get('pred')).format())
    }).evaluate(function (r) {
      readout.setValue('TRUE (point, ' + year + '): ' + (r.t || '(no point here)') +
                       '\nPREDICTED (MapBiomas): ' + (r.p || 'no data'));
    });
  });
  
  // Save data 
  Export.table.toDrive({
  collection: res,
  description: 'points_vs_mapbiomas_' + year,
  fileFormat: 'CSV'
});
}