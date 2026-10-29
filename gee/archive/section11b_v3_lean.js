// ARCHIVED (superseded): lean 11b with its own small SCL-masked collection; replaced by task B of 02_qc_error_analysis_export.js, which exports the same NDVI per polygon as a CSV.

// ===== 11b (v3, lean). NDVI per year at each Cropin QC polygon =====
// Replace the whole current 11b block (from "===== 11b" down to just before "===== 11c") with this.
// Why the earlier versions failed: they reused the big Sentinel-2 collection from section 5,
// which is joined to Cloud Score+ across all three states – too heavy. This builds a small,
// separate collection and uses Sentinel-2's own cloud flag (SCL band) instead.

var qcSpots = qc.map(function (f) {               // a 15 m circle at the centre of each polygon
  return f.setGeometry(f.geometry().centroid(1).buffer(15));
});
var box = qc.geometry().bounds(1);

function yearlyNDVI(y) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(box)
    .filterDate(y + '-01-01', (y + 1) + '-01-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .map(function (img) {
      var scl = img.select('SCL');                  // 4 = vegetation, 5 = bare soil (keep), clouds dropped
      var clear = scl.eq(4).or(scl.eq(5));
      return img.normalizedDifference(['B8', 'B4']).rename('NDVI').updateMask(clear);
    });
  return ee.Image.cat(
    col.median().rename('med_' + y),                               // typical greenness that year
    col.reduce(ee.Reducer.percentile([10])).rename('p10_' + y));   // low point → harvest/clearing dips
}

var years = [2019, 2020, 2021, 2022, 2023, 2024];
var yearly = ee.Image.cat(years.map(yearlyNDVI));
var vals = yearly.reduceRegions({collection: qcSpots, reducer: ee.Reducer.mean(), scale: 10, tileScale: 16});

var medBands = years.map(function (y) { return 'med_' + y; });
var p10Bands = years.map(function (y) { return 'p10_' + y; });
var summary = ee.FeatureCollection(['plantation', 'forest'].map(function (k) {
  var s = vals.filter(ee.Filter.eq('qc_label', k));
  var props = {label: k};
  medBands.concat(p10Bands).forEach(function (b) { props[b] = s.aggregate_mean(b); });
  return ee.Feature(null, props);
}));
print('11b yearly NDVI averages by group:', summary);
print(ui.Chart.feature.byProperty({features: summary, xProperties: p10Bands, seriesProperty: 'label'})
  .setChartType('LineChart')
  .setOptions({title: '11b Low-point (10th pct) NDVI per year – dips = harvest/clearing',
               colors: ['d68a00', '1f8d49'], vAxis: {title: 'NDVI'}, pointSize: 4}));

['plantation', 'forest'].forEach(function (k) {
  var s = vals.filter(ee.Filter.eq('qc_label', k)).filter(ee.Filter.notNull(['p10_2019', 'p10_2020']));
  var dipped = s.filter(ee.Filter.or(ee.Filter.lt('p10_2019', 0.5), ee.Filter.lt('p10_2020', 0.5)));
  print('11b ' + k.toUpperCase() + ': share with a dip (NDVI < 0.5) in 2019–2020:',
        dipped.size().divide(s.size()));
});

// If the Console STILL says "memory limit exceeded", remove the "//" from the next two lines and Run.
// It runs the same calculation as a background task (more memory) and saves a CSV to your Google Drive:
// go to the Tasks tab → click RUN next to "qc_ndvi_by_year". Send that CSV to Claude to chart.
// Export.table.toDrive({collection: vals.select(['qc_label'].concat(medBands, p10Bands)),
//                       description: 'qc_ndvi_by_year', fileFormat: 'CSV'});
