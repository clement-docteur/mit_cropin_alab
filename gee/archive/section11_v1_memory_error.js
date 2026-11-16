// ARCHIVED (superseded): first section 11; 11b computed monthly NDVI over the big Cloud Score+ collection and hit 'User memory limit exceeded'.

// ---------- 11. Missed plantations vs. real forest: compare ALL polygons at once ----------
// Paste this at the very bottom of your script (below section 10) and press Run.
// Results appear in the Console; charts may take ~1 minute.

// The two groups we care about (both called "forest" by MapBiomas 2020):
var missed = qcMB.filter(ee.Filter.and(
  ee.Filter.eq('qc_label', 'plantation'), ee.Filter.eq('mode', 3)));   // ~282 plantations MapBiomas missed
var realForest = qcMB.filter(ee.Filter.and(
  ee.Filter.eq('qc_label', 'forest'), ee.Filter.eq('mode', 3)));       // ~120 real forests
var both = missed.merge(realForest);

// ===== 11a. CANOPY HEIGHT: how tall, and how uniform? =====
// Hypothesis: plantations are planted at the same time, so heights are more even (lower std. dev.)
// We look at a 30 m circle around each polygon so there are enough 10 m pixels to measure spread.
var circles = both.map(function (f) {
  return f.setGeometry(f.geometry().centroid(1).buffer(30));
});
var hStats = height.rename('h').reduceRegions({
  collection: circles,
  reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true),
  scale: 10
});
['plantation', 'forest'].forEach(function (k) {
  var s = hStats.filter(ee.Filter.eq('qc_label', k));
  print('11a ' + k.toUpperCase() + ' (n=', s.size(), ')',
        'avg canopy height (m):', s.aggregate_mean('mean'),
        'avg height variability, std dev (m):', s.aggregate_mean('stdDev'));
});
print(ui.Chart.feature.histogram(hStats.filter(ee.Filter.eq('qc_label', 'plantation')), 'mean', 20)
  .setOptions({title: '11a Canopy height – missed PLANTATIONS', colors: ['d68a00']}));
print(ui.Chart.feature.histogram(hStats.filter(ee.Filter.eq('qc_label', 'forest')), 'mean', 20)
  .setOptions({title: '11a Canopy height – real FOREST', colors: ['1f8d49']}));

// ===== 11b. NDVI OVER TIME: average monthly greenness for each group, 2019–2024 =====
// Hypothesis: plantations show harvest dips / regrowth; forest stays steady until it's cleared.
var empty = ee.Image(0).float().rename('NDVI').selfMask();   // placeholder for cloudy months
var start = ee.Date('2019-01-01');
var ndviHere = ndvi.filterBounds(both.geometry());
var monthly = ee.ImageCollection(ee.List.sequence(0, 71).map(function (m) {
  var d = start.advance(m, 'month');
  return ndviHere.filterDate(d, d.advance(1, 'month'))
    .merge(ee.ImageCollection([empty])).median()
    .set('system:time_start', d.millis());
}));
var groups = ee.FeatureCollection([
  ee.Feature(missed.geometry(), {label: 'Plantation (missed by MapBiomas)'}),
  ee.Feature(realForest.geometry(), {label: 'Real forest'})
]);
print(ui.Chart.image.seriesByRegion({
  imageCollection: monthly, regions: groups, reducer: ee.Reducer.mean(),
  band: 'NDVI', scale: 10, xProperty: 'system:time_start', seriesProperty: 'label'
}).setOptions({title: '11b Average monthly NDVI', colors: ['d68a00', '1f8d49'],
               interpolateNulls: true, vAxis: {title: 'NDVI'}}));

// Share of polygons with a big greenness dip BEFORE 2021 (i.e. an earlier harvest)
var minPre2021 = monthly.filterDate('2019-01-01', '2021-01-01').min().rename('minNDVI');
var dips = minPre2021.reduceRegions({collection: both, reducer: ee.Reducer.mean(), scale: 10})
  .filter(ee.Filter.notNull(['mean']));
['plantation', 'forest'].forEach(function (k) {
  var s = dips.filter(ee.Filter.eq('qc_label', k));
  print('11b ' + k.toUpperCase() + ': share with NDVI below 0.5 at some point in 2019–2020:',
        s.filter(ee.Filter.lt('mean', 0.5)).size().divide(s.size()));
});

// ===== 11c. LOOK AT EXAMPLES on satellite imagery =====
// Change these two settings and press Run to jump between examples.
var showForest = false;  // false = a missed plantation, true = a real forest
var pick = 0;            // 0, 1, 2, 3 … picks a different polygon
var pool = showForest ? realForest : missed;
var ex = ee.Feature(pool.toList(1, pick).get(0));
Map.addLayer(ex, {color: 'yellow'}, 'Selected example');
Map.centerObject(ex, 17);
Map.setOptions('SATELLITE');
print('11c Example – Cropin label:', ex.get('qc_label'), '(now click inside the yellow outline)');
