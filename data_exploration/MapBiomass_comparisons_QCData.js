// =============================================================================
// QC data (2026) vs MapBiomas predictions (2025) - forest vs plantation
// QC data is small, so everything computes live - no export step needed.
// =============================================================================

// ---- SETTINGS ---------------------------------------------------------------
var PRED_YEAR    = 2025;   // latest MapBiomas year (QC data is from 2026)
var FOREST_CODES = [3];    // MapBiomas codes counted as forest
var PLANT_CODES  = [9];    // MapBiomas codes counted as plantation (9 = silviculture)
var RUN_CHARTS   = true;   // false = skip the exploration charts (faster)

var qc        = ee.FeatureCollection('projects/cropin-510901/assets/cropin_brazil_qc_forest_plantation_1');
var QC_LABEL  = 'qc_label';     // QC column holding the label
var QC_FOREST = 'forest';       // value in that column meaning forest
var QC_PLANT  = 'plantation';   // value in that column meaning plantation
// -----------------------------------------------------------------------------

var GREEN = '1f8d49', ORANGE = 'ff7f00', CYAN = '00e5ff', MAGENTA = 'ff00ff';
var names   = ee.Dictionary({'0': 'other', '1': 'forest', '2': 'plantation'});
var colors  = ee.Dictionary({'0': '999999', '1': GREEN, '2': ORANGE});
var jsNames = {0: 'other', 1: 'forest', 2: 'plantation'};

print('QC example feature (check QC_LABEL and its values)');
print(qc.first());

// ---- MapBiomas prediction, consolidated: 1 forest, 2 plantation, 0 other ------
var lulc = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1')
  .select('classification_' + PRED_YEAR);
var fromCodes = FOREST_CODES.concat(PLANT_CODES);
var toCodes = FOREST_CODES.map(function () { return 1; })
  .concat(PLANT_CODES.map(function () { return 2; }));
var pred = lulc.remap(fromCodes, toCodes, 0).rename('pred');

function qcToClass(v) {
  return ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(v, QC_FOREST), 1,
                   ee.Algorithms.If(ee.Algorithms.IsEqual(v, QC_PLANT), 2, 0)));
}

// ---- QC vs prediction: most common predicted class inside each QC feature ---
var cmp = pred.reduceRegions({
  collection: qc, reducer: ee.Reducer.mode(), scale: 10, tileScale: 4
}).map(function (f) {
  var t = qcToClass(f.get(QC_LABEL));
  var p = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(f.get('mode'), null), 0, f.get('mode'))).int();
  return f.set({true_code: t, pred_code: p, agree: t.eq(p),
                QC: names.get(t.int().format()), PREDICTED: names.get(p.int().format())});
});

// ---- Layers (bottom to top) - toggle each in the Layers menu ----------------
var predLayer = ui.Map.Layer(pred.selfMask(), {min: 1, max: 2, palette: [GREEN, ORANGE]},
                             'MapBiomas predicted (' + PRED_YEAR + ')', true, 0.7);
var qcLayer = ui.Map.Layer(cmp.map(function (f) {
  return f.set('style', {color: '000000', width: 1, pointSize: 4,
    fillColor: ee.String(colors.get(ee.Number(f.get('true_code')).int().format())).cat('CC')});
}).style({styleProperty: 'style'}), {}, 'QC data (2026)', true);
Map.layers().add(predLayer);
Map.layers().add(qcLayer);

Map.addLayer(cmp.filter(ee.Filter.eq('agree', 1))
               .style({color: CYAN, fillColor: CYAN + '40', width: 2, pointSize: 5}),
             {}, 'AGREES with MapBiomas', false);
Map.addLayer(cmp.filter(ee.Filter.eq('agree', 0))
               .style({color: MAGENTA, fillColor: MAGENTA + '80', width: 2, pointSize: 5}),
             {}, 'DISAGREES with MapBiomas', false);

// ---- Confusion matrix -------------------------------------------------------
var cm = cmp.errorMatrix('true_code', 'pred_code', [0, 1, 2]);
print('Confusion matrix: rows TRUE (QC), cols PREDICTED (other / forest / plantation)');
print(cm);
print('Overall accuracy');
print(cm.accuracy());

Map.centerObject(qc);

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
  if (qcLayer.getShown()) {
    var q = qc.filterBounds(near).first();
    req.qc = ee.Algorithms.If(q, names.get(qcToClass(ee.Feature(q).get(QC_LABEL)).int().format()), null);
  }
  if (predLayer.getShown()) {
    req.pred = pred.reduceRegion(ee.Reducer.first(), pt, 10).get('pred');
  }
  if (Object.keys(req).length === 0) { caption.setValue('Turn on a layer to see its value'); return; }
  caption.setValue('Loading...');
  ee.Dictionary(req).evaluate(function (r) {
    var lines = [];
    if ('qc' in req)   lines.push('QC Value: ' + (r.qc || '(no QC feature here)'));
    if ('pred' in req) lines.push('Predicted: ' + (jsNames[r.pred] || 'no data'));
    caption.setValue(lines.join('\n'));
  });
});

// =============================================================================
// EXPLORE THE MISMATCHES - charts appear in the Console (allow ~1 minute)
// Groups (QC label vs MapBiomas prediction):
//   missed plantation = QC plantation, MapBiomas says forest   <- the false positives
//   real forest       = QC forest,     MapBiomas says forest   <- the comparison group
//   caught plantation = QC plantation, MapBiomas says plantation
// =============================================================================
if (RUN_CHARTS) {
  var GROUPS = ['missed plantation', 'real forest', 'caught plantation'];

  var grouped = cmp.map(function (f) {
    var t = ee.Number(f.get('true_code')), p = ee.Number(f.get('pred_code'));
    var g = ee.Algorithms.If(t.eq(2).and(p.eq(1)), GROUPS[0],
            ee.Algorithms.If(t.eq(1).and(p.eq(1)), GROUPS[1],
            ee.Algorithms.If(t.eq(2).and(p.eq(2)), GROUPS[2], 'other mismatch')));
    return f.set({group: g, area_ha: f.geometry().area(1).divide(1e4)});
  });
  print('Polygons per group');
  print(grouped.aggregate_histogram('group'));

  // ---- Chart 1: what MapBiomas actually calls each QC polygon (full legend) ---
  var classNames = ee.Dictionary({
    '3': 'Forest', '4': 'Savanna', '5': 'Mangrove', '6': 'Floodable forest', '9': 'Silviculture',
    '11': 'Wetland', '12': 'Grassland', '15': 'Pasture', '18': 'Agriculture', '19': 'Temporary crop',
    '21': 'Mosaic of uses', '24': 'Urban', '25': 'Other non-veg.', '29': 'Rocky outcrop',
    '33': 'Water', '36': 'Perennial crop', '49': 'Wooded restinga', '77': 'Herb.-shrub'
  });
  var raw = lulc.rename('code').reduceRegions({
    collection: qc, reducer: ee.Reducer.mode(), scale: 10, tileScale: 4
  }).filter(ee.Filter.notNull(['mode']));
  var histP = ee.Dictionary(raw.filter(ee.Filter.eq(QC_LABEL, QC_PLANT)).aggregate_histogram('mode'));
  var histF = ee.Dictionary(raw.filter(ee.Filter.eq(QC_LABEL, QC_FOREST)).aggregate_histogram('mode'));
  var classFc = ee.FeatureCollection(histP.keys().cat(histF.keys()).distinct().map(function (k) {
    var code = ee.Number.parse(k).int().format();
    return ee.Feature(null, {
      cls: code.cat(' ').cat(classNames.get(code, 'other')),
      'QC plantation': histP.get(k, 0), 'QC forest': histF.get(k, 0)
    });
  }));
  print(ui.Chart.feature.byFeature({features: classFc, xProperty: 'cls',
                                    yProperties: ['QC plantation', 'QC forest']})
    .setChartType('ColumnChart')
    .setOptions({title: '1. What MapBiomas ' + PRED_YEAR + ' calls the QC polygons',
                 colors: ['d68a00', '1f8d49'],
                 hAxis: {title: 'MapBiomas class (most common inside polygon)'},
                 vAxis: {title: 'Number of polygons'}}));

  // ---- Features per polygon: canopy height + NDVI over the year (one pass) ----
  var box = qc.geometry().bounds(1);
  var ndviCol = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')   // small, local collection
    .filterBounds(box)
    .filterDate(PRED_YEAR + '-01-01', (PRED_YEAR + 1) + '-01-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .map(function (img) {
      var scl = img.select('SCL');                       // keep vegetation (4) + bare soil (5)
      return img.normalizedDifference(['B8', 'B4']).rename('NDVI')
                .updateMask(scl.eq(4).or(scl.eq(5)));
    });
  var stack = ee.Image.cat(
    ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').rename('height'), // 2020 heights
    ndviCol.median().rename('ndvi_med'),                              // typical greenness
    ndviCol.reduce(ee.Reducer.percentile([10])).rename('ndvi_p10'),   // low point: harvest / clearing
    ndviCol.reduce(ee.Reducer.stdDev()).rename('ndvi_sd')             // seasonality
  );

  // 30 m circle at each polygon's centre -> enough 10 m pixels to measure spread
  var spots = grouped.filter(ee.Filter.inList('group', GROUPS)).map(function (f) {
  var g = f.geometry();
  // big enough polygons: shrink 10 m inward to avoid edges; tiny ones: small centre circle
  return f.setGeometry(ee.Geometry(ee.Algorithms.If(
    g.area(1).gt(2000), g.buffer(-10, 1), g.centroid(1).buffer(15))));
});  // -> height_mean, height_stdDev, ndvi_med_mean, ndvi_p10_mean, ndvi_sd_mean, ...

  // ---- Chart 2: average of each feature per group (table) ---------------------
  var metrics = ['area_ha', 'height_mean', 'height_stdDev',
                 'ndvi_med_mean', 'ndvi_p10_mean', 'ndvi_sd_mean'];
  var summary = ee.FeatureCollection(GROUPS.map(function (g) {
    var s = feats.filter(ee.Filter.eq('group', g));
    var props = {group: g, n: s.size()};
    metrics.forEach(function (m) { props[m] = s.aggregate_mean(m); });
    return ee.Feature(null, props);
  }));
  print(ui.Chart.feature.byFeature({features: summary, xProperty: 'group',
                                    yProperties: ['n'].concat(metrics)})
    .setChartType('Table')
    .setOptions({title: '2. Group averages'}));
  print('Key: area_ha = polygon size; height = canopy height (m, 2020) and height_stdDev = how uneven; ' +
        'ndvi_med = typical greenness; ndvi_p10 = low point (dips = harvest/clearing); ' +
        'ndvi_sd = how much greenness swings over ' + PRED_YEAR);

  // ---- Charts 3-4: does any pair of features separate the groups? -------------
  var clean = feats.filter(ee.Filter.notNull(['height_mean', 'height_stdDev',
                                              'ndvi_p10_mean', 'ndvi_sd_mean']));
  print(ui.Chart.feature.groups({features: clean, xProperty: 'height_mean',
                                 yProperty: 'height_stdDev', seriesProperty: 'group'})
    .setChartType('ScatterChart')
    .setOptions({title: '3. Canopy height vs. unevenness (plantations expected: more even)',
                 hAxis: {title: 'Average canopy height (m)'},
                 vAxis: {title: 'Height std dev (m)'}, pointSize: 3}));
  print(ui.Chart.feature.groups({features: clean, xProperty: 'ndvi_p10_mean',
                                 yProperty: 'ndvi_sd_mean', seriesProperty: 'group'})
    .setChartType('ScatterChart')
    .setOptions({title: '4. NDVI low point vs. NDVI swing (plantations expected: deeper dips)',
                 hAxis: {title: 'NDVI 10th percentile'},
                 vAxis: {title: 'NDVI std dev over the year'}, pointSize: 3}));

  // ---- Optional: send everything to Google Drive as a CSV for Python/R --------
  // Export.table.toDrive({collection: feats.select(['group', QC_LABEL].concat(metrics)),
  //                       description: 'qc_mismatch_features_' + PRED_YEAR, fileFormat: 'CSV'});
}
