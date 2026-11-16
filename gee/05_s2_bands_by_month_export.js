// #####################################################################
// 05_s2_bands_by_month_export.js  (Meera, Nov 2026)
// WHAT IT DOES
//   Deepak (11/4 meeting) asked us to look beyond SAG and NDVI at the full set of Sentinel-2
//   bands, especially SWIR (B11, B12), and to compare Dec-Jan with the other months. This
//   script builds, for each calendar month, a median composite of every Sentinel-2 band and
//   reads it at the 2,290 MapBiomas reference points and at the 433 QC polygons. The notebook
//   then scores every band x month cell (how well does it separate plantation from forest?).
// ASSETS NEEDED (GEE project cropin-510901)
//   projects/cropin-510901/assets/mapbiomas_ref_south  (2,290 points: id, label, state, high_conf)
//   projects/cropin-510901/assets/cropin_qc            (433 polygons: id, qc_label)
//   Public: Sentinel-2 SR harmonized, Cloud Score+.
// HOW TO RUN
//   Paste into a new script in the Code Editor (project cropin-510901) and press Run. In the
//   Tasks tab, click RUN on all 4 tasks (reference points split by state, plus the QC set).
//   Files land in Google Drive/cropin_exports. Copy them to data/gee_exports/.
// OUTPUTS (one row per point or polygon, no dates)
//   ref_s2_bands_by_month_{PR,SC,RS}.csv: id, label, state, high_conf, then the columns below
//   qc_s2_bands_by_month.csv:             id, qc_label, then the columns below
//   <band>_<MM>  median surface reflectance (x 10,000) of that band in calendar month MM
//                (01 = January ... 12 = December), pooled over 2019-2022, averaged over the
//                point's 15 m buffer or over the polygon. Bands: B2 blue, B3 green, B4 red,
//                B5/B6/B7 red edge, B8 NIR, B8A narrow NIR, B11 SWIR-1, B12 SWIR-2.
//   SAG_<MM>     median SAG (degrees) in that month, same formula and dark-pixel rule as
//                gee/03 and gee/04, so the grid has Cropin's index as a benchmark row
//   n_<MM>       average number of clear observations per pixel behind that month's median
// CHOICES
//   - Years 2019-2022: the reference labels are stable in 2016, 2020 and 2022, and the QC
//     clearings happened in 2026 (most Apr-Sep), so this window is before the clearing for both
//     sets. Pooling four years gives each month enough clear images (cloudy winters).
//   - Same cloud handling as gee/03/04: CLOUDY_PIXEL_PERCENTAGE < 90, Cloud Score+ cs >= 0.6.
//   - 20 m bands (B5-B7, B8A, B11, B12) are read at 10 m (nearest pixel), as GEE does by default.
//   - B1 (aerosol) and B9 (water vapour) are atmospheric bands and are left out; B10 is not in
//     the surface-reflectance product.
// #####################################################################

var PROJECT = 'cropin-510901';
var ref = ee.FeatureCollection('projects/' + PROJECT + '/assets/mapbiomas_ref_south');
var qc = ee.FeatureCollection('projects/' + PROJECT + '/assets/cropin_qc');
var BUFFER_M = 15;
var STATES = ['PR', 'SC', 'RS'];
var BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
var START = '2019-01-01', END = '2023-01-01';   // end is exclusive

var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var csBands = csPlus.first().bandNames();

function addSAG(img) {                      // identical to gee/03 and gee/04 (Cropin's addGreenAngle())
  var yB = img.select('B2').divide(10000), yG = img.select('B3').divide(10000), yR = img.select('B4').divide(10000);
  var xB = 0.4924, xG = 0.5598, xR = 0.6646;  // band centre wavelengths (µm)
  var dx1 = xB - xG, dy1 = yB.subtract(yG);   // green → blue
  var dx2 = xR - xG, dy2 = yR.subtract(yG);   // green → red
  var dot = dy1.multiply(dy2).add(dx1 * dx2);
  var m1 = dy1.pow(2).add(dx1 * dx1).sqrt(), m2 = dy2.pow(2).add(dx2 * dx2).sqrt();
  var sag = dot.divide(m1.multiply(m2)).clamp(-1, 1).acos().multiply(180 / Math.PI).rename('SAG');
  var dark = yG.lte(0.05).and(yR.lte(0.05)).and(yB.lte(0.05));
  return sag.updateMask(dark);
}

function pad(m) { return (m < 10 ? '0' : '') + m; }

// One image with a band per (band, month): B2_01 ... B12_12, SAG_01 ... SAG_12, n_01 ... n_12
function monthlyStack(region) {
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region)
    .filterDate(START, END)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
    .linkCollection(csPlus, csBands)
    .map(function (img) { return img.updateMask(img.select('cs').gte(0.6)); });

  var parts = [];
  for (var m = 1; m <= 12; m++) {
    var mm = pad(m);
    var col = s2.filter(ee.Filter.calendarRange(m, m, 'month'));
    parts.push(col.select(BANDS).median()
      .rename(BANDS.map(function (b) { return b + '_' + mm; })));
    parts.push(col.map(addSAG).median().rename('SAG_' + mm));
    parts.push(col.select('B4').count().rename('n_' + mm));
  }
  return ee.Image.cat(parts);
}

var VALUE_COLS = [];
for (var m = 1; m <= 12; m++) {
  BANDS.concat(['SAG', 'n']).forEach(function (b) { VALUE_COLS.push(b + '_' + pad(m)); });
}

// --- Reference points, one task per state ---
STATES.forEach(function (st) {
  var pts = ref.filter(ee.Filter.eq('state', st))
    .map(function (f) { return f.buffer(BUFFER_M); });
  var table = monthlyStack(pts).reduceRegions({
      collection: pts, reducer: ee.Reducer.mean(), scale: 10, tileScale: 4})
    .map(function (f) { return f.setGeometry(null); });
  Export.table.toDrive({
    collection: table,
    description: 'ref_s2_bands_by_month_' + st,
    folder: 'cropin_exports',
    fileFormat: 'CSV',
    selectors: ['id', 'label', 'state', 'high_conf'].concat(VALUE_COLS)
  });
});

// --- QC polygons ---
var qcTable = monthlyStack(qc).reduceRegions({
    collection: qc, reducer: ee.Reducer.mean(), scale: 10, tileScale: 4})
  .map(function (f) { return f.setGeometry(null); });
Export.table.toDrive({
  collection: qcTable,
  description: 'qc_s2_bands_by_month',
  folder: 'cropin_exports',
  fileFormat: 'CSV',
  selectors: ['id', 'qc_label'].concat(VALUE_COLS)
});

print('Columns per row:', VALUE_COLS.length + ' values (10 bands + SAG + n, x 12 months)');
print('Four export tasks ready. Open Tasks and click RUN on each.');
