// =============================================================================
// Late-winter NDVI dip, 2019-2022: missed plantations vs real forest vs caught plantations
// For each point and year:  peak = median NDVI Mar-Jun,  dip = median NDVI Aug-Oct,
//                           drop = peak - dip  (bigger = stronger late-winter loss of greenness)
//
// HOW TO RUN (two phases):
//   1) Leave EXPORTED = false and run. Check the point count in the Console, then
//      in the Tasks tab click RUN on the export task and wait for it to finish.
//   2) Set EXPORTED = true and run again: the yearly charts load from the saved result.
// =============================================================================

// ---- SETTINGS ---------------------------------------------------------------
var YEARS        = [2019, 2020, 2021, 2022];  // Sentinel-2 SR coverage of Brazil is patchy before late 2018
var LABEL_YEAR   = 2022;   // groups are defined by this year's label + MapBiomas prediction
var STABLE_FROM  = 2019;   // keep only points with the same label from this year to LABEL_YEAR
var FOREST_CODES = [3];    // MapBiomas codes counted as forest
var PLANT_CODES  = [9];    // MapBiomas codes counted as plantation (9 = silviculture)
var SOUTH_ONLY   = true;   // true = only Parana, Santa Catarina, Rio Grande do Sul
var RADIUS       = 15;     // metres around each point (15 m = ~7 pixels)
var PEAK_MONTHS  = [3, 6]; // Mar-Jun (inclusive)
var DIP_MONTHS   = [7, 11];// Jul-Nov (inclusive) - wide, so early and late dips are both caught
var EXPORTED     = false;  // flip to true after the export task finishes

var points = ee.FeatureCollection('projects/cropin-510901/assets/mapbiomas_85k_points_validation');
var RESULT = 'projects/cropin-510901/assets/val_winter_dip_' + YEARS[0] + '_' + YEARS[YEARS.length - 1];
// -----------------------------------------------------------------------------

var GROUPS  = ['missed plantation', 'real forest', 'caught plantation'];
var GCOLORS = ['d68a00', '1f8d49', '7a5900'];   // orange, green, dark brown

// ---- MapBiomas prediction (LABEL_YEAR), consolidated: 1 forest, 2 plantation, 0 other
var lulc = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1')
  .select('classification_' + LABEL_YEAR);
var fromCodes = FOREST_CODES.concat(PLANT_CODES);
var toCodes = FOREST_CODES.map(function () { return 1; })
  .concat(PLANT_CODES.map(function () { return 2; }));
var pred = lulc.remap(fromCodes, toCodes, 0).rename('pred');

// ---- Validation points: forest/plantation, no edges, label stable -----------
var truthCol = 'CLASS_' + LABEL_YEAR;
var pts = points.filter(ee.Filter.eq('BORDA_' + LABEL_YEAR, 0))
                .filter(ee.Filter.inList(truthCol, fromCodes));
if (SOUTH_ONLY) {
  pts = pts.filter(ee.Filter.rangeContains('CD_MUN', 4100000, 4399999)); // IBGE codes 41-43
}
var stableCols = [];
for (var y = STABLE_FROM; y <= LABEL_YEAR; y++) { stableCols.push('CLASS_' + y); }
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
  // PHASE 1: group each point, then measure peak / dip / drop for every year
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
  var empty = ee.Image.constant(0).rename('NDVI').updateMask(0);

  function windowMedian(col, year, months) {
    var w = col.filterDate(ee.Date.fromYMD(year, months[0], 1),
                           ee.Date.fromYMD(year, months[1], 1).advance(1, 'month'));
    return ee.Image(ee.Algorithms.If(w.size().gt(0), w.median(), empty));
  }

  var yearImgs = YEARS.map(function (yr) {
    var ndvi = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(box)
      .filterDate(ee.Date.fromYMD(yr, PEAK_MONTHS[0], 1),
                  ee.Date.fromYMD(yr, DIP_MONTHS[1], 1).advance(1, 'month'))
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
      .map(function (img) {
        var scl = img.select('SCL');                   // keep vegetation (4) + bare soil (5)
        return img.normalizedDifference(['B8', 'B4']).rename('NDVI')
                  .updateMask(scl.eq(4).or(scl.eq(5)))
                  .copyProperties(img, ['system:time_start']);   // keep date for window filters
      });
        var peak = windowMedian(ndvi, yr, PEAK_MONTHS);

    // Low point = 10th percentile of ALL clear images Jul-Nov: close to the lowest
    // value whenever the dip happens, but ignores one-off cloud outliers
    var w = ndvi.filterDate(ee.Date.fromYMD(yr, DIP_MONTHS[0], 1),
                            ee.Date.fromYMD(yr, DIP_MONTHS[1], 1).advance(1, 'month'));
    var dip = ee.Image(ee.Algorithms.If(w.size().gt(0),
      w.reduce(ee.Reducer.percentile([10])).rename('NDVI'), empty));

    // Month of the lowest monthly NDVI (7 = Jul ... 11 = Nov)
    var monthly = [];
    for (var m = DIP_MONTHS[0]; m <= DIP_MONTHS[1]; m++) {
      var med = windowMedian(ndvi, yr, [m, m]);
      monthly.push(med.multiply(-1).rename('q')            // lowest NDVI = highest q
        .addBands(ee.Image.constant(m).toFloat().rename('month').updateMask(med.mask())));
    }
    var minMonth = ee.ImageCollection(monthly).qualityMosaic('q').select('month');

    return ee.Image.cat(peak, dip, peak.subtract(dip), minMonth)
             .rename(['peak_' + yr, 'dip_' + yr, 'drop_' + yr, 'minmonth_' + yr]);
  });
  var stack = ee.Image.cat(yearImgs);   // 3 bands x number of years

  var feats = stack.reduceRegions({
    collection: sampled.map(function (f) { return f.setGeometry(f.geometry().buffer(RADIUS)); }),
    reducer: ee.Reducer.mean(), scale: 10, tileScale: 16
  }).map(function (f) { return f.setGeometry(f.geometry().centroid(1)); });

  Export.table.toAsset({collection: feats,
                        description: 'val_winter_dip_' + YEARS[0] + '_' + YEARS[YEARS.length - 1],
                        assetId: RESULT});
  print('Phase 1: open the Tasks tab, click RUN, then set EXPORTED = true and rerun.');

} else {
  // ===========================================================================
  // PHASE 2: yearly medians per group -> two line charts (drop, dip)
  // ===========================================================================
  var res = ee.FeatureCollection(RESULT);
  print('Points per group (labels stable ' + STABLE_FROM + '-' + LABEL_YEAR + ')');
  print(res.aggregate_histogram('group'));

    var CHARTS = [
    {key: 'drop', title: '1. Late-winter NDVI drop (Mar-Jun peak minus Jul-Nov low point)',
     yTitle: 'NDVI drop (bigger = more loss of greenness)'},
    {key: 'dip',  title: '2. Late-winter NDVI low point (10th percentile, Jul-Nov)',
     yTitle: 'NDVI'},
    {key: 'minmonth', title: '3. Month of lowest NDVI (7 = Jul ... 11 = Nov)',
     yTitle: 'Month'}
  ];
  var NODATA = -9999;

  // All medians fetched in ONE request
  var reqs = {hist: res.aggregate_histogram('group')};
  CHARTS.forEach(function (c) {
    YEARS.forEach(function (yr, i) {
      var band = c.key + '_' + yr;
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
    var keep = [];
    GROUPS.forEach(function (g, j) {
      var n = (r.hist && r.hist[g]) || 0;
      if (n > 0) { keep.push(j); } else { print('Note: no points in group "' + g + '".'); }
      if (j === 0 && n > 0 && n < 20) {
        print('Warning: only ' + n + ' missed plantations - medians will be noisy. ' +
              'Consider STABLE_FROM = 2020 and re-exporting.');
      }
    });

    CHARTS.forEach(function (c) {
      // Table built by hand with value columns typed as numbers (avoids the
      // "cannot be of type string" chart error when a year has no data)
      var cols = [{id: 'year', label: 'Year', type: 'string'}];
      keep.forEach(function (j) { cols.push({id: 'g' + j, label: GROUPS[j], type: 'number'}); });
      var rows = YEARS.map(function (yr, i) {
        var cells = [{v: String(yr)}];
        keep.forEach(function (j) {
          var v = r[c.key + '_' + i + '_' + j];
          cells.push({v: (v === NODATA || v === null || v === undefined) ? null : v});
        });
        return {c: cells};
      });
      print(ui.Chart({cols: cols, rows: rows}, 'LineChart', {
        title: c.title,
        colors: keep.map(function (j) { return GCOLORS[j]; }),
        hAxis: {title: 'Year'}, vAxis: {title: c.yTitle},
        pointSize: 5, lineWidth: 2, interpolateNulls: true
      }));
    });
    print('Reading the charts: if orange (missed) and dark brown (caught plantation) stay ' +
          'above green (forest) in chart 1 every year, the late-winter drop is a consistent signal.');
  });
  
    // ---- Overlap check: how many real forests look like a typical missed plantation?
  // Threshold = the missed plantations' median, so by design ~half of them pass it.
  var missedPts = res.filter(ee.Filter.eq('group', 'missed plantation'));
  var forestPts = res.filter(ee.Filter.eq('group', 'real forest'));
  function r3(x) { return ee.Number(x).multiply(1000).round().divide(1000); }
  var overlap = ee.FeatureCollection(YEARS.map(function (yr) {
    var dropB = 'drop_' + yr, dipB = 'dip_' + yr;
    var mD = missedPts.filter(ee.Filter.notNull([dropB]));
    var mL = missedPts.filter(ee.Filter.notNull([dipB]));
    var fD = forestPts.filter(ee.Filter.notNull([dropB]));
    var fL = forestPts.filter(ee.Filter.notNull([dipB]));
    var tDrop = ee.Number(mD.reduceColumns(ee.Reducer.median(), [dropB]).get('median'));
    var tDip  = ee.Number(mL.reduceColumns(ee.Reducer.median(), [dipB]).get('median'));
    var fDropN = fD.filter(ee.Filter.gte(dropB, tDrop)).size();
    var fDipN  = fL.filter(ee.Filter.lte(dipB, tDip)).size();
    return ee.Feature(null, {
      'Year': String(yr),
      'Drop threshold': r3(tDrop),
      'Missed >= (n)': mD.filter(ee.Filter.gte(dropB, tDrop)).size(),
      'Forest >= (n)': fDropN,
      'Forest >= (%)': fDropN.divide(fD.size()).multiply(100).round(),
      'Low-point threshold': r3(tDip),
      'Missed <= (n)': mL.filter(ee.Filter.lte(dipB, tDip)).size(),
      'Forest <= (n)': fDipN,
      'Forest <= (%)': fDipN.divide(fL.size()).multiply(100).round()
    });
  }));
  print(ui.Chart.feature.byFeature({
    features: overlap, xProperty: 'Year',
    yProperties: ['Drop threshold', 'Missed >= (n)', 'Forest >= (n)', 'Forest >= (%)',
                  'Low-point threshold', 'Missed <= (n)', 'Forest <= (n)', 'Forest <= (%)']
  }).setChartType('Table').setOptions({title: 'Overlap check'}));
  print('Overlap check: a rule set at the missed plantations\' median catches the "Missed" ' +
        'count and wrongly flags the "Forest" count. Compare the two counts, not just the %: ' +
        'forest points far outnumber missed plantations.');

  // Map: points coloured by group
  GROUPS.forEach(function (g, i) {
    Map.addLayer(res.filter(ee.Filter.eq('group', g)).style({color: GCOLORS[i], pointSize: 4}),
                 {}, 'Points: ' + g);
  });

  // ---- Optional: CSV to Google Drive for Python/R -----------------------------
  // Export.table.toDrive({collection: res, description: 'val_winter_dip', fileFormat: 'CSV'});
}

if (SOUTH_ONLY) { Map.setCenter(-51.5, -27.5, 6); } else { Map.setCenter(-51.0, -15.0, 4); }