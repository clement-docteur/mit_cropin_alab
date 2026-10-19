// =============================================================================
// Validation points vs MapBiomas predictions - 2022 - forest vs plantation
// + exploration charts for the mismatches (canopy height, NDVI)
//
// HOW TO RUN (two phases, so nothing times out):
//   1) Leave EXPORTED = false and run. In the Tasks tab, click RUN on the
//      export task and wait for it to finish (it computes the comparison AND
//      the per-point features in one go).
//   2) Set EXPORTED = true and run again: agree/disagree layers, confusion
//      matrix and charts all load quickly from the saved result.
// =============================================================================

// ---- SETTINGS ---------------------------------------------------------------
var YEAR         = 2022;   // same year for labels and predictions (points end in 2022)
var FOREST_CODES = [3];    // MapBiomas codes counted as forest
var PLANT_CODES  = [9];    // MapBiomas codes counted as plantation (9 = silviculture)
var SOUTH_ONLY   = true;   // true = only Parana, Santa Catarina, Rio Grande do Sul
var EXPORTED     = true;  // flip to true after the export task finishes
var RUN_CHARTS   = true;   // false = skip the exploration charts (faster)
var RADIUS       = 15;     // metres around each point used for features (15 m = ~7 pixels)

var points = ee.FeatureCollection('projects/cropin-510901/assets/mapbiomas_85k_points_validation');
var RESULT = 'projects/cropin-510901/assets/val_vs_mapbiomas_features_' + YEAR;
// -----------------------------------------------------------------------------

var GREEN = '1f8d49', ORANGE = 'ff7f00', CYAN = '00e5ff', MAGENTA = 'ff00ff';
var names   = ee.Dictionary({'0': 'other', '1': 'forest', '2': 'plantation'});
var colors  = ee.Dictionary({'0': '999999', '1': GREEN, '2': ORANGE});
var jsNames = {0: 'other', 1: 'forest', 2: 'plantation'};
var GROUPS  = ['missed plantation', 'real forest', 'caught plantation'];

// ---- MapBiomas prediction, consolidated: 1 forest, 2 plantation, 0 other ------
var lulc = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1')
  .select('classification_' + YEAR);
var fromCodes = FOREST_CODES.concat(PLANT_CODES);
var toCodes = FOREST_CODES.map(function () { return 1; })
  .concat(PLANT_CODES.map(function () { return 2; }));
var pred = lulc.remap(fromCodes, toCodes, 0).rename('pred');

// ---- Validation points (forest/plantation only, edge points removed) --------
var truthCol = 'CLASS_' + YEAR;
var pts = points.filter(ee.Filter.eq('BORDA_' + YEAR, 0));
if (SOUTH_ONLY) {
  pts = pts.filter(ee.Filter.rangeContains('CD_MUN', 4100000, 4399999)); // IBGE codes 41-43
}
pts = pts.filter(ee.Filter.inList(truthCol, fromCodes)).map(function (f) {
  var v = ee.Number(f.get(truthCol));
  var c = ee.Number(ee.Algorithms.If(ee.List(FOREST_CODES).contains(v), 1, 2)).int();
  return ee.Feature(f.geometry(), {true_code: c, VAL: names.get(c.format())});
});

// ---- Layers (bottom to top) - toggle each in the Layers menu ----------------
var predLayer = ui.Map.Layer(pred.selfMask(), {min: 1, max: 2, palette: [GREEN, ORANGE]},
                             'MapBiomas predicted (' + YEAR + ')', true, 0.7);
var valLayer = ui.Map.Layer(pts.map(function (f) {
  return f.set('style', {color: '000000', width: 1, pointSize: 4,
    fillColor: ee.String(colors.get(ee.Number(f.get('true_code')).int().format())).cat('CC')});
}).style({styleProperty: 'style'}), {}, 'Validation points (' + YEAR + ')', true);
Map.layers().add(predLayer);
Map.layers().add(valLayer);

if (!EXPORTED) {
  // ===========================================================================
  // PHASE 1: comparison + features per point, saved once as an asset
  // ===========================================================================
  // 1a. MapBiomas at each point: consolidated class + raw legend code
  var sampled = pred.addBands(lulc.rename('code')).sampleRegions({
    collection: pts, properties: ['true_code', 'VAL'],
    scale: 10, geometries: true, tileScale: 16
  }).map(function (f) {
    var t = ee.Number(f.get('true_code')).int(), p = ee.Number(f.get('pred')).int();
    var g = ee.Algorithms.If(t.eq(2).and(p.eq(1)), GROUPS[0],
            ee.Algorithms.If(t.eq(1).and(p.eq(1)), GROUPS[1],
            ee.Algorithms.If(t.eq(2).and(p.eq(2)), GROUPS[2], 'other mismatch')));
    return f.set({pred_code: p, PREDICTED: names.get(p.format()), agree: t.eq(p),
                  code: ee.Number(f.get('code')).int(), group: g});
  });

  // 1b. Features: canopy height (2020, nearest available) + NDVI over YEAR
  var box = pts.geometry().bounds(1);
  var ndviCol = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(box)
    .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .map(function (img) {
      var scl = img.select('SCL');                       // keep vegetation (4) + bare soil (5)
      return img.normalizedDifference(['B8', 'B4']).rename('NDVI')
                .updateMask(scl.eq(4).or(scl.eq(5)));
    });
  var stack = ee.Image.cat(
    ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').rename('height'),
    ndviCol.median().rename('ndvi_med'),                              // typical greenness
    ndviCol.reduce(ee.Reducer.percentile([10])).rename('ndvi_p10'),   // low point: harvest / clearing
    ndviCol.reduce(ee.Reducer.stdDev()).rename('ndvi_sd')             // seasonality
  );

  // Small circle around each point (RADIUS) -> a few 10 m pixels to measure spread
  // while staying away from neighbouring stands; edge points were already dropped
  // with the BORDA filter. Geometry is set back to the point afterwards.
  var feats = stack.reduceRegions({
    collection: sampled.map(function (f) { return f.setGeometry(f.geometry().buffer(RADIUS)); }),
    reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true),
    scale: 10, tileScale: 16
  }).map(function (f) { return f.setGeometry(f.geometry().centroid(1)); });
  // -> height_mean, height_stdDev, ndvi_med_mean, ndvi_p10_mean, ndvi_sd_mean, ...

  Export.table.toAsset({collection: feats, description: 'val_vs_mapbiomas_features_' + YEAR,
                        assetId: RESULT});
  print('Phase 1: open the Tasks tab, click RUN, then set EXPORTED = true and rerun.');

} else {
  // ===========================================================================
  // PHASE 2: load the saved result (fast)
  // ===========================================================================
  var res = ee.FeatureCollection(RESULT);
  Map.addLayer(res.filter(ee.Filter.eq('agree', 1)).style({color: CYAN, pointSize: 4}),
               {}, 'AGREES with MapBiomas', false);
  Map.addLayer(res.filter(ee.Filter.eq('agree', 0)).style({color: MAGENTA, pointSize: 5}),
               {}, 'DISAGREES with MapBiomas', false);

  var cm = res.errorMatrix('true_code', 'pred_code', [0, 1, 2]);
  print('Confusion matrix: rows TRUE, cols PREDICTED (other / forest / plantation)');
  print(cm);
  print('Overall accuracy');
  print(cm.accuracy());

  if (RUN_CHARTS) {
    print('Points per group');
    print(res.aggregate_histogram('group'));

    // ---- Chart 1: what MapBiomas calls each point (full legend) -------------
    var classNames = ee.Dictionary({
      '3': 'Forest', '4': 'Savanna', '5': 'Mangrove', '6': 'Floodable forest', '9': 'Silviculture',
      '11': 'Wetland', '12': 'Grassland', '15': 'Pasture', '18': 'Agriculture', '19': 'Temporary crop',
      '21': 'Mosaic of uses', '24': 'Urban', '25': 'Other non-veg.', '29': 'Rocky outcrop',
      '33': 'Water', '36': 'Perennial crop', '49': 'Wooded restinga', '77': 'Herb.-shrub'
    });
    var histP = ee.Dictionary(res.filter(ee.Filter.eq('true_code', 2)).aggregate_histogram('code'));
    var histF = ee.Dictionary(res.filter(ee.Filter.eq('true_code', 1)).aggregate_histogram('code'));
    var classFc = ee.FeatureCollection(histP.keys().cat(histF.keys()).distinct().map(function (k) {
      var code = ee.Number.parse(k).int().format();
      return ee.Feature(null, {
        cls: code.cat(' ').cat(classNames.get(code, 'other')),
        'True plantation': histP.get(k, 0), 'True forest': histF.get(k, 0)
      });
    }));
    print(ui.Chart.feature.byFeature({features: classFc, xProperty: 'cls',
                                      yProperties: ['True plantation', 'True forest']})
      .setChartType('ColumnChart')
      .setOptions({title: '1. What MapBiomas ' + YEAR + ' calls the validation points',
                   colors: ['d68a00', '1f8d49'],
                   hAxis: {title: 'MapBiomas class at the point'},
                   vAxis: {title: 'Number of points'}}));

    // ---- Chart 2: average of each feature per group (table) -----------------
    var metrics = ['height_mean', 'height_stdDev', 'ndvi_med_mean', 'ndvi_p10_mean', 'ndvi_sd_mean'];
    var summary = ee.FeatureCollection(GROUPS.map(function (g) {
      var s = res.filter(ee.Filter.eq('group', g));
      var props = {group: g, n: s.size()};
      // MEDIAN per group: not dragged around by outliers the way a mean is
      metrics.forEach(function (m) {
        props[m] = s.filter(ee.Filter.notNull([m]))
                    .reduceColumns(ee.Reducer.median(), [m]).get('median');
      });
      return ee.Feature(null, props);
    }));
    print(ui.Chart.feature.byFeature({features: summary, xProperty: 'group',
                                      yProperties: ['n'].concat(metrics)})
      .setChartType('Table')
      .setOptions({title: '2. Group medians'}));
    print('Key: height = canopy height (m, 2020) and height_stdDev = how uneven; ' +
          'ndvi_med = typical greenness; ndvi_p10 = low point (dips = harvest/clearing); ' +
          'ndvi_sd = how much greenness swings over ' + YEAR);

    // ---- Charts 3-4: scatter plots ------------------------------------------
    // One column per group, so colours are FIXED in both charts:
    // missed plantation = orange, real forest = green, caught plantation = dark brown
    var clean = res.filter(ee.Filter.inList('group', GROUPS))
      .filter(ee.Filter.notNull(['height_mean', 'height_stdDev', 'ndvi_p10_mean', 'ndvi_sd_mean']))
      .randomColumn('r', 1).sort('r').limit(4000);     // charts cap out around 5,000 points

    function scatter(xProp, yProp, title, xTitle, yTitle) {
      var fc = clean.map(function (f) {
        var g = f.get('group'), y = f.get(yProp), props = {};
        GROUPS.forEach(function (name) {
          props[name] = ee.Algorithms.If(ee.Algorithms.IsEqual(g, name), y, null);
        });
        return f.set(props);
      });
      print(ui.Chart.feature.byFeature({features: fc, xProperty: xProp, yProperties: GROUPS})
        .setChartType('ScatterChart')
        .setOptions({title: title, colors: ['d68a00', '1f8d49', '7a5900'],
                     hAxis: {title: xTitle}, vAxis: {title: yTitle}, pointSize: 3}));
    }
    scatter('height_mean', 'height_stdDev', '3. Canopy height vs. unevenness',
            'Average canopy height (m)', 'Height std dev (m)');
    scatter('ndvi_p10_mean', 'ndvi_sd_mean', '4. NDVI low point vs. NDVI swing',
            'NDVI 10th percentile', 'NDVI std dev over the year');

    // ---- Optional: CSV to Google Drive for Python/R ---------------------------
    // Export.table.toDrive({collection: res.select(['group', 'true_code', 'pred_code', 'code']
    //                         .concat(metrics)), description: 'val_features_' + YEAR,
    //                       fileFormat: 'CSV'});
  }
}

if (SOUTH_ONLY) { Map.setCenter(-51.5, -27.5, 6); } else { Map.setCenter(-51.0, -15.0, 4); }

// ---- Legend -----------------------------------------------------------------
function legendRow(color, text) {
  return ui.Panel([
    ui.Label('', {backgroundColor: '#' + color, padding: '8px', margin: '2px 6px 2px 0'}),
    ui.Label(text, {margin: '2px 0'})
  ], ui.Panel.Layout.Flow('horizontal'));
}
var legend = ui.Panel({style: {position: 'bottom-left', padding: '6px 10px'}});
legend.add(ui.Label('Legend', {fontWeight: 'bold'}));
legend.add(legendRow(GREEN, 'Forest'));
legend.add(legendRow(ORANGE, 'Plantation'));
legend.add(legendRow(CYAN, 'Agrees with MapBiomas'));
legend.add(legendRow(MAGENTA, 'Disagrees with MapBiomas'));
Map.add(legend);

// ---- Click caption: one line per layer that is switched on ------------------
var caption = ui.Label('Click the map', {position: 'top-center', whiteSpace: 'pre'});
Map.add(caption);
Map.style().set('cursor', 'crosshair');

Map.onClick(function (coords) {
  var pt = ee.Geometry.Point(coords.lon, coords.lat);
  var near = pt.buffer(Math.max(30, Map.getScale() * 6)); // ~6 screen pixels at any zoom
  var req = {};
  if (valLayer.getShown()) {
    var v = pts.filterBounds(near).first();
    req.val = ee.Algorithms.If(v, ee.Feature(v).get('VAL'), null);
  }
  if (predLayer.getShown()) {
    req.pred = pred.reduceRegion(ee.Reducer.first(), pt, 10).get('pred');
  }
  if (Object.keys(req).length === 0) { caption.setValue('Turn on a layer to see its value'); return; }
  caption.setValue('Loading...');
  ee.Dictionary(req).evaluate(function (r) {
    var lines = [];
    if ('val' in req)  lines.push('Validation Value: ' + (r.val || '(no point here)'));
    if ('pred' in req) lines.push('Predicted: ' + (jsNames[r.pred] || 'no data'));
    caption.setValue(lines.join('\n'));
  });
});