// =============================================================================
// Seasonal profiles 2022: missed plantations vs real forest vs caught plantations
// Monthly NDVI, NDMI (Sentinel-2) and radar VH, VH/VV (Sentinel-1) per group
//
// HOW TO RUN (two phases - 12 months x thousands of points is too heavy live):
//   1) Leave EXPORTED = false and run. In the Tasks tab, click RUN on the
//      export task and wait for it to finish.
//   2) Set EXPORTED = true and run again: the monthly charts load from the
//      saved result.
// =============================================================================

// ---- SETTINGS ---------------------------------------------------------------
var YEAR         = 2022;   // label year = imagery year
var STABLE_FROM  = 2020;   // keep only points with the same label from this year to YEAR
var FOREST_CODES = [3];    // MapBiomas codes counted as forest
var PLANT_CODES  = [9];    // MapBiomas codes counted as plantation (9 = silviculture)
var SOUTH_ONLY   = true;   // true = only Parana, Santa Catarina, Rio Grande do Sul
var RADIUS       = 15;     // metres around each point (15 m = ~7 pixels)
var EXPORTED     = false;  // flip to true after the export task finishes

var points = ee.FeatureCollection('projects/cropin-510901/assets/mapbiomas_85k_points_validation');
var RESULT = 'projects/cropin-510901/assets/val_monthly_' + YEAR;
// -----------------------------------------------------------------------------

var GROUPS  = ['missed plantation', 'real forest', 'caught plantation'];
var GCOLORS = ['d68a00', '1f8d49', '7a5900'];   // orange, green, dark brown
var MONTHS  = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
function pad(m) { return ('0' + m).slice(-2); }

// ---- MapBiomas prediction, consolidated: 1 forest, 2 plantation, 0 other ------
var lulc = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1')
  .select('classification_' + YEAR);
var fromCodes = FOREST_CODES.concat(PLANT_CODES);
var toCodes = FOREST_CODES.map(function () { return 1; })
  .concat(PLANT_CODES.map(function () { return 2; }));
var pred = lulc.remap(fromCodes, toCodes, 0).rename('pred');

// ---- Validation points: forest/plantation, no edges, label stable -----------
var truthCol = 'CLASS_' + YEAR;
var pts = points.filter(ee.Filter.eq('BORDA_' + YEAR, 0))
                .filter(ee.Filter.inList(truthCol, fromCodes));
if (SOUTH_ONLY) {
  pts = pts.filter(ee.Filter.rangeContains('CD_MUN', 4100000, 4399999)); // IBGE codes 41-43
}
// Same label every year STABLE_FROM..YEAR: collect the labels, keep if only 1 distinct
var stableCols = [];
for (var y = STABLE_FROM; y <= YEAR; y++) { stableCols.push('CLASS_' + y); }
pts = pts.map(function (f) {
  var labels = ee.List(stableCols.map(function (c) { return f.get(c); }));
  return f.set('stable', labels.distinct().size().eq(1));
}).filter(ee.Filter.eq('stable', 1));

print('Points left after filters (must be > 0):');
print(pts.size());

pts = pts.map(function (f) {
  var v = ee.Number(f.get(truthCol));
  var c = ee.Number(ee.Algorithms.If(ee.List(FOREST_CODES).contains(v), 1, 2)).int();
  return ee.Feature(f.geometry(), {true_code: c});
});

if (!EXPORTED) {
  // ===========================================================================
  // PHASE 1: group each point, then measure 12 months of features
  // ===========================================================================
  var sampled = pred.sampleRegions({
    collection: pts, properties: ['true_code'], scale: 10, geometries: true, tileScale: 16
  }).map(function (f) {
    var t = ee.Number(f.get('true_code')).int(), p = ee.Number(f.get('pred')).int();
    var g = ee.Algorithms.If(t.eq(2).and(p.eq(1)), GROUPS[0],
            ee.Algorithms.If(t.eq(1).and(p.eq(1)), GROUPS[1],
            ee.Algorithms.If(t.eq(2).and(p.eq(2)), GROUPS[2], 'other mismatch')));
    return f.set({pred_code: p, group: g});
  }).filter(ee.Filter.inList('group', GROUPS));

  var box = pts.geometry().bounds(1);

  // Sentinel-2: NDVI (greenness) + NDMI (canopy moisture), cloud-masked with SCL
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(box)
    .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .map(function (img) {
      var scl = img.select('SCL');                     // keep vegetation (4) + bare soil (5)
      return ee.Image.cat(img.normalizedDifference(['B8', 'B4']).rename('NDVI'),
                          img.normalizedDifference(['B8', 'B11']).rename('NDMI'))
                              .updateMask(scl.eq(4).or(scl.eq(5)))
               .copyProperties(img, ['system:time_start']);   // keep the date for monthly filtering
    });

  // Sentinel-1 radar (already in dB): VV, VH; VH - VV in dB = VH/VV ratio
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(box)
    .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);

  var emptyS2 = ee.Image.constant([0, 0]).rename(['NDVI', 'NDMI']).updateMask(0);
  var emptyS1 = ee.Image.constant([0, 0]).rename(['VV', 'VH']).updateMask(0);

  var monthImgs = [];
  for (var m = 1; m <= 12; m++) {
    var start = ee.Date.fromYMD(YEAR, m, 1), end = start.advance(1, 'month');
    var s2m = s2.filterDate(start, end), s1m = s1.filterDate(start, end);
    var opt = ee.Image(ee.Algorithms.If(s2m.size().gt(0), s2m.median(), emptyS2));
    var rad = ee.Image(ee.Algorithms.If(s1m.size().gt(0), s1m.median(), emptyS1));
    monthImgs.push(ee.Image.cat(
      opt.select('NDVI'), opt.select('NDMI'), rad.select('VV'), rad.select('VH'),
      rad.select('VH').subtract(rad.select('VV'))
    ).rename(['ndvi_' + pad(m), 'ndmi_' + pad(m), 'vv_' + pad(m),
              'vh_' + pad(m), 'ratio_' + pad(m)]));
  }
  var stack = ee.Image.cat(monthImgs);   // 60 bands: 5 features x 12 months

  var feats = stack.reduceRegions({
    collection: sampled.map(function (f) { return f.setGeometry(f.geometry().buffer(RADIUS)); }),
    reducer: ee.Reducer.mean(), scale: 10, tileScale: 16
  }).map(function (f) { return f.setGeometry(f.geometry().centroid(1)); });

  Export.table.toAsset({collection: feats, description: 'val_monthly_' + YEAR, assetId: RESULT});
  print('Phase 1: open the Tasks tab, click RUN, then set EXPORTED = true and rerun.');

} else {
  // ===========================================================================
  // PHASE 2: monthly medians per group -> line charts
  // ===========================================================================
  var res = ee.FeatureCollection(RESULT);
  print('Points per group (labels stable ' + STABLE_FROM + '-' + YEAR + ')');
  print(res.aggregate_histogram('group'));

    // ---- Monthly medians per group, fetched in ONE request ----------------------
  // The chart table is built by hand with every value column typed as 'number',
  // so months with no clear images (nulls) can't confuse the chart library.
  var CHARTS = [
    {key: 'ndvi',  title: '1. NDVI (greenness)',              yTitle: 'NDVI'},
    {key: 'ndmi',  title: '2. NDMI (canopy moisture)',        yTitle: 'NDMI'},
    {key: 'vh',    title: '3. Radar VH (canopy volume)',      yTitle: 'VH backscatter (dB)'},
    {key: 'ratio', title: '4. Radar VH/VV ratio (structure)', yTitle: 'VH - VV (dB)'}
  ];
  var NODATA = -9999;
  var reqs = {hist: res.aggregate_histogram('group')};
  CHARTS.forEach(function (c) {
    MONTHS.forEach(function (name, i) {
      var band = c.key + '_' + pad(i + 1);
      GROUPS.forEach(function (g, j) {
        var med = res.filter(ee.Filter.eq('group', g)).filter(ee.Filter.notNull([band]))
                     .reduceColumns(ee.Reducer.median(), [band]).get('median');
        reqs[c.key + '_' + i + '_' + j] =
          ee.Algorithms.If(ee.Algorithms.IsEqual(med, null), NODATA, med);
      });
    });
  });

  ee.Dictionary(reqs).evaluate(function (r, err) {
    if (err) { print('Chart data error: ' + err); return; }
    var keep = [];                                   // groups that have points
    GROUPS.forEach(function (g, j) {
      if (r.hist && r.hist[g] > 0) { keep.push(j); }
      else { print('Note: no points in group "' + g + '", so it is left out of the charts.'); }
    });

    CHARTS.forEach(function (c) {
      var cols = [{id: 'month', label: 'Month', type: 'string'}];
      keep.forEach(function (j) { cols.push({id: 'g' + j, label: GROUPS[j], type: 'number'}); });
      var rows = MONTHS.map(function (name, i) {
        var cells = [{v: name}];
        keep.forEach(function (j) {
          var v = r[c.key + '_' + i + '_' + j];
          cells.push({v: (v === NODATA || v === null || v === undefined) ? null : v});
        });
        return {c: cells};
      });
      print(ui.Chart({cols: cols, rows: rows}, 'LineChart', {
        title: c.title + ' - median per group, ' + YEAR,
        colors: keep.map(function (j) { return GCOLORS[j]; }),
        hAxis: {title: 'Month (southern winter = Jun-Aug)'},
        vAxis: {title: c.yTitle}, pointSize: 4, lineWidth: 2, interpolateNulls: true
      }));
    });
    print('Reading the charts: where the orange (missed plantation) and green (real forest) ' +
          'lines separate, and in which months, is a seasonal feature that could tell them apart.');
  });
  // Map: points coloured by group, on top of the MapBiomas prediction
  Map.addLayer(pred.selfMask(), {min: 1, max: 2, palette: ['1f8d49', 'ff7f00']},
               'MapBiomas predicted (' + YEAR + ')', true, 0.6);
  GROUPS.forEach(function (g, i) {
    Map.addLayer(res.filter(ee.Filter.eq('group', g)).style({color: GCOLORS[i], pointSize: 4}),
                 {}, 'Points: ' + g);
  });

  // ---- Optional: CSV to Google Drive for Python/R -----------------------------
  // Export.table.toDrive({collection: res, description: 'val_monthly_' + YEAR,
  //                       fileFormat: 'CSV'});
}

if (SOUTH_ONLY) { Map.setCenter(-51.5, -27.5, 6); } else { Map.setCenter(-51.0, -15.0, 4); }