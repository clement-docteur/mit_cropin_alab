// #####################################################################
// 04_sag_mapbiomas_ref_export.js  (Meera, Nov 2026)
// WHAT IT DOES
//   Out-of-sample check of SAG. Builds the same SAG time series as gee/03 (TASK D), but on the
//   2,290 MapBiomas reference points in PR/SC/RS (1,762 forest, 528 plantation; class stable in
//   2016, 2020 and 2022) instead of the 433 QC polygons. The labels come from MapBiomas'
//   human-interpreted validation points, so this test does not depend on Cropin's QC set.
// ASSETS NEEDED (GEE project cropin-510901)
//   projects/cropin-510901/assets/mapbiomas_ref_south  (2,290 points: id, label, state,
//   high_conf, edge2020, agree2020, ...; built by src/build_mapbiomas_ref_south.py)
//   Public: Sentinel-2 SR harmonized, Cloud Score+.
// HOW TO RUN
//   Paste into a new script in the Code Editor (project cropin-510901) and press Run. In the
//   Tasks tab, click RUN on all 3 tasks (one per state, so no single task gets too big). Files
//   land in Google Drive/cropin_exports. Copy them to data/gee_exports/.
// OUTPUTS
//   ref_sag_timeseries_PR.csv, _SC.csv, _RS.csv: one row per (point, Sentinel-2 image),
//   Jan 2019 to Jan 2026, all months. Columns: id, label, state, high_conf, date, tile,
//   mean (SAG in degrees over the valid pixels), count (valid pixels).
// SAME AS gee/03 (so QC and reference numbers are comparable)
//   SAG formula, dark-pixel rule (B2, B3, B4 all <= 0.05), Cloud Score+ cs >= 0.6 via
//   updateMask, CLOUDY_PIXEL_PERCENTAGE < 90, dates 2019-01-01 to 2026-02-01, scale 10 m.
// DIFFERENT FROM gee/03 (deliberate)
//   - Points, not polygons. Each point is buffered by 15 m, i.e. about a 3x3 block of 10 m
//     pixels, close to the size of a QC polygon (about 3 pixels) and to the 30 m Landsat pixel
//     the MapBiomas interpreters labelled. A single pixel would lose many dates to the
//     dark-pixel rule.
//   - Overlapping S2 tiles still give two rows per (id, date). Average them per date in Python,
//     as in notebook 02.
// CAVEAT
//   Labels describe 2016-2022. A plantation harvested in 2023-25 will look different in later
//   seasons, so the analysis should also be run on the seasons up to 2022/23 only.
// #####################################################################

var PROJECT = 'cropin-510901';
var ref = ee.FeatureCollection('projects/' + PROJECT + '/assets/mapbiomas_ref_south');
var BUFFER_M = 15;
var STATES = ['PR', 'SC', 'RS'];

print('Reference points:', ref.size(), ref.aggregate_histogram('label'));

var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var csBands = csPlus.first().bandNames();

function addSAG(img) {                      // identical to gee/03 (Cropin's addGreenAngle())
  var yB = img.select('B2').divide(10000), yG = img.select('B3').divide(10000), yR = img.select('B4').divide(10000);
  var xB = 0.4924, xG = 0.5598, xR = 0.6646;  // band centre wavelengths (µm)
  var dx1 = xB - xG, dy1 = yB.subtract(yG);   // green → blue
  var dx2 = xR - xG, dy2 = yR.subtract(yG);   // green → red
  var dot = dy1.multiply(dy2).add(dx1 * dx2);
  var m1 = dy1.pow(2).add(dx1 * dx1).sqrt(), m2 = dy2.pow(2).add(dx2 * dx2).sqrt();
  var sag = dot.divide(m1.multiply(m2)).clamp(-1, 1).acos().multiply(180 / Math.PI).rename('SAG');
  var dark = yG.lte(0.05).and(yR.lte(0.05)).and(yB.lte(0.05));
  return sag.updateMask(dark)
    .set('system:time_start', img.get('system:time_start'))
    .set('tile', img.get('MGRS_TILE'));
}

STATES.forEach(function (st) {
  var pts = ref.filter(ee.Filter.eq('state', st))
    .map(function (f) { return f.buffer(BUFFER_M); });

  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(pts)
    .filterDate('2019-01-01', '2026-02-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
    .linkCollection(csPlus, csBands)
    .map(function (img) { return img.updateMask(img.select('cs').gte(0.6)); })
    .map(addSAG);

  var table = s2.map(function (img) {
    var fc = pts.filterBounds(img.geometry());
    var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
    return img.reduceRegions({
        collection: fc,
        reducer: ee.Reducer.mean().combine(ee.Reducer.count(), '', true),
        scale: 10, tileScale: 4})
      .filter(ee.Filter.gt('count', 0))
      .map(function (f) { return f.set({date: date, tile: img.get('tile')}).setGeometry(null); });
  }).flatten();

  Export.table.toDrive({
    collection: table,
    description: 'ref_sag_timeseries_' + st,
    folder: 'cropin_exports',
    fileFormat: 'CSV',
    selectors: ['id', 'label', 'state', 'high_conf', 'date', 'tile', 'mean', 'count']
  });
});

print('Three export tasks ready (ref_sag_timeseries_PR/SC/RS). Open Tasks and click RUN on each.');
