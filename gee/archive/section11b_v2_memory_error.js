// ARCHIVED (superseded): lighter 11b (15 m circles, yearly summaries) still reused the heavy section-5 collection and hit the memory limit. Still embedded in 01.

// ===== 11b (FIXED, lighter). NDVI over time: yearly summaries, 2019–2024 =====
// Replace the whole old "11b" block (from "===== 11b" down to just before "===== 11c") with this.

// One small circle per polygon (15 m radius ≈ a few 10 m pixels)
var spots = both.map(function (f) {
  return f.setGeometry(f.geometry().centroid(1).buffer(15));
});
var area = spots.geometry().bounds();   // simple box instead of a complex union → much less memory

// For each year: median NDVI (typical greenness) and 10th-percentile NDVI (catches harvest dips)
var years = [2019, 2020, 2021, 2022, 2023, 2024];
var yearly = ee.Image.cat(years.map(function (y) {
  var col = ndvi.filterBounds(area).filterDate(y + '-01-01', (y + 1) + '-01-01');
  return ee.Image.cat(
    col.median().rename('med_' + y),
    col.reduce(ee.Reducer.percentile([10])).rename('p10_' + y));
}));

// One pass over all polygons
var vals = yearly.reduceRegions({collection: spots, reducer: ee.Reducer.mean(), scale: 10, tileScale: 4});

// Average per group → a small table you can chart
var medBands = years.map(function (y) { return 'med_' + y; });
var p10Bands = years.map(function (y) { return 'p10_' + y; });
var summary = ee.FeatureCollection(['plantation', 'forest'].map(function (k) {
  var s = vals.filter(ee.Filter.eq('qc_label', k));
  var props = {label: k};
  medBands.concat(p10Bands).forEach(function (b) { props[b] = s.aggregate_mean(b); });
  return ee.Feature(null, props);
}));
print('11b yearly NDVI averages by group:', summary);

print(ui.Chart.feature.byProperty({features: summary, xProperties: medBands, seriesProperty: 'label'})
  .setChartType('LineChart')
  .setOptions({title: '11b Typical (median) NDVI per year', colors: ['d68a00', '1f8d49'],
               vAxis: {title: 'NDVI'}, pointSize: 4}));
print(ui.Chart.feature.byProperty({features: summary, xProperties: p10Bands, seriesProperty: 'label'})
  .setChartType('LineChart')
  .setOptions({title: '11b Low-point (10th percentile) NDVI per year – dips = harvest/clearing',
               colors: ['d68a00', '1f8d49'], vAxis: {title: 'NDVI'}, pointSize: 4}));

// Share of polygons with a clear dip in 2019 or 2020 (before Cropin's alerts)
['plantation', 'forest'].forEach(function (k) {
  var s = vals.filter(ee.Filter.eq('qc_label', k))
              .filter(ee.Filter.notNull(['p10_2019', 'p10_2020']));
  var dipped = s.filter(ee.Filter.or(ee.Filter.lt('p10_2019', 0.5), ee.Filter.lt('p10_2020', 0.5)));
  print('11b ' + k.toUpperCase() + ': share with a dip (NDVI < 0.5) in 2019–2020:',
        dipped.size().divide(s.size()));
});
