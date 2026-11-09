// =============================================================================
// Harvest cycles from LandTrendr (Landsat 1985 -> CUTOFF) for plantation vs natural forest
//
// Per 30 m Landsat pixel, using ONLY imagery up to 31 Dec of the cutoff year (EUDR: 2020):
//   n_cuts            number of abrupt NBR drops (clear-cuts), from LandTrendr segments  -> plantation (risky)
//   n_cycles          cuts followed by regrowth of >= REC_FRAC of the drop within MAX_REC_YEARS, measured
//                     over the next one or two LandTrendr segments                       -> plantation (risky)
//   last_cut_year     year of the most recent cut (0 = none)
//   max_drop          largest cut, in NBR units
//   forest_since_1985 no cut AND at most STABLE_TOLERANCE yearly values below FOREST_LEVEL    -> natural (veto)
//   abrupt_start      a year at forest level with >= MIN_OPEN_YEARS open years (<= OPEN_LEVEL) in the 8 years
//                     before it and the last open year at most FAST_RISE_YEARS earlier       -> plantation
//   years_below_forest, n_vertices (diagnostics)
// plus the yearly NBR values (nbr_1985 ... nbr_CUTOFF) and Hansen GFC, so the features can be recomputed
// offline with harvest_cycles/scripts/features_from_gee_export.py (exactly the code of the analysis).
//
// Differences from the offline analysis (harvest_cycles/docs/summary.md), so compare via the adapter:
//   - Collection 2 surface reflectance (offline: Collection 1 top of atmosphere) -> recalibrate levels.
//   - Real LandTrendr (offline: a LandTrendr-like dynamic-programming fit chosen by BIC).
//   - forest_since_1985 / abrupt_start use raw yearly values (offline: after one-year spike removal),
//     and abrupt_start uses an 8-year window ending the year before forest level is reached.
//   - Overlapping rows of the same path are not de-duplicated (offline they are).
//
// HOW TO RUN
//   1) MODE = 'CALIBRATE' -> Tasks tab: run 'harvest_calibration_<CUTOFF>'. The CSV gives
//      closed_p5 = 5th percentile of all 2000-CUTOFF yearly NBR values of closed-canopy pixels (Hansen
//      tree cover >= 90% in 2000, no loss up to CUTOFF, no gain) and open_p25 = 25th percentile of the
//      1999-2001 values of open pixels (tree cover <= 10% in 2000, no gain), in the three states: the
//      same statistics as the offline analysis. Set FOREST_LEVEL = closed_p5 and
//      OPEN_LEVEL = min(open_p25, closed_p5 - DROP / 2). Surface-reflectance NBR is higher than the
//      top-of-atmosphere NBR used offline, so do not reuse the offline numbers.
//   2) MODE = 'MAP'    -> look at the feature layers (zoom in on a few plantations and forests).
//   3) MODE = 'EXPORT' -> Tasks tab: run the two CSV exports (validation points, QC polygons), then
//      python harvest_cycles/scripts/features_from_gee_export.py <val.csv> <qc.csv> --out features_gee.csv
// =============================================================================

// ---- SETTINGS ---------------------------------------------------------------
var MODE         = 'CALIBRATE';   // 'CALIBRATE' | 'MAP' | 'EXPORT'
var START_YEAR   = 1985;
var CUTOFF       = 2020;          // last year of imagery used (EUDR cutoff)
var MONTHS       = [1, 12];       // months used for each yearly value (all year = robust default);
                                  // [4, 9] = dry season (clearer, but deciduous forest is leafless)
var HARMONISE    = true;          // Landsat 8 OLI -> ETM+ (Roy et al. 2016, RMA coefficients)

// Thresholds, in NBR units (surface reflectance). Set FOREST_LEVEL / OPEN_LEVEL from CALIBRATE.
var DROP             = 0.20;  // minimum NBR fall of a cut
var MAX_CUT_YEARS    = 3;     // a cut must happen within this many years (abrupt)
var REC_FRAC         = 0.5;   // regrowth must recover this share of the drop to count as a cycle
var MAX_REC_YEARS    = 10;    // ... within this many years
var FOREST_LEVEL     = 0.70;  // PLACEHOLDER - replace with CALIBRATE closed_p5
var OPEN_LEVEL       = 0.55;  // PLACEHOLDER - replace with min(open_p25, closed_p5 - DROP / 2)
var STABLE_TOLERANCE = 0;     // yearly values allowed below FOREST_LEVEL for forest_since_1985
var MIN_OPEN_YEARS   = 3;     // open years needed in the 8 years before an 'abrupt start'
var FAST_RISE_YEARS  = 5;     // last open year -> forest level within this many years = abrupt

// LandTrendr. maxSegments: each rotation needs ~3 segments (cut, regrowth, plateau); the usual 6 caps a
// pixel at ~2 cuts, so multi-rotation eucalyptus (up to ~5 cycles since 1985) needs more. The offline
// analysis uses 18 (12 is a cheaper compromise for state-wide maps; check the run time).
// recoveryThreshold: the default 0.25 forbids any regrowth faster than 4 years, which erases eucalyptus
// coppice cycles (2-4 years to canopy). 1.0 only forbids 1-year recoveries.
var LT_PARAMS = {
  maxSegments: 18, spikeThreshold: 0.9, vertexCountOvershoot: 3, preventOneYearRecovery: true,
  recoveryThreshold: 1.0, pvalThreshold: 0.05, bestModelProportion: 0.75, minObservationsNeeded: 6
};

var points = ee.FeatureCollection('projects/cropin-510901/assets/mapbiomas_85k_points_validation');
var qc     = ee.FeatureCollection('projects/cropin-510901/assets/cropin_brazil_qc_forest_plantation_1');
var hansen = ee.Image('UMD/hansen/global_forest_change_2025_v1_13');
// -----------------------------------------------------------------------------

// Validation points in Parana, Santa Catarina, Rio Grande do Sul (IBGE codes 41-43)
var val3 = points.filter(ee.Filter.rangeContains('CD_MUN', 4100000, 4399999));
// The three states: the GAUL level-1 units that contain those points (avoids spelling issues)
var states = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Brazil'))
  .filterBounds(val3.randomColumn('r', 3).sort('r').limit(300).geometry());
var REGION = ee.Geometry.Rectangle([-58.0, -34.0, -48.0, -22.4]);   // for filterBounds only

// Output column names (client-side, for the CSV selectors)
var FEATURE_NAMES = ['n_cuts', 'n_cycles', 'last_cut_year', 'max_drop', 'forest_since_1985', 'abrupt_start',
                     'years_below_forest', 'n_vertices'];
var YEARS_JS = [], YEAR_NAMES = [];
for (var yy = START_YEAR; yy <= CUTOFF; yy++) { YEARS_JS.push(yy); YEAR_NAMES.push('nbr_' + yy); }
var NY = YEARS_JS.length;
var HANSEN_NAMES = ['hansen_tc2000', 'hansen_gain', 'hansen_lossyear', 'hansen_loss_le_cutoff'];
var OUT_NAMES = FEATURE_NAMES.concat(YEAR_NAMES, HANSEN_NAMES);

// ---- Landsat Collection 2 Level 2: cloud-masked NBR per image ---------------
function nbrImage(img, nirBand, swirBand, isOli) {
  var qa = img.select('QA_PIXEL');
  var clear = qa.bitwiseAnd(1 << 0).eq(0)          // fill
    .and(qa.bitwiseAnd(1 << 1).eq(0))              // dilated cloud
    .and(qa.bitwiseAnd(1 << 3).eq(0))              // cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0))              // cloud shadow
    .and(qa.bitwiseAnd(1 << 5).eq(0));             // snow
  if (isOli) { clear = clear.and(qa.bitwiseAnd(1 << 2).eq(0)); }   // cirrus (OLI only)
  clear = clear.and(img.select('QA_RADSAT').eq(0));
  var sr = img.select([nirBand, swirBand], ['nir', 'swir2']).multiply(0.0000275).add(-0.2);
  if (isOli && HARMONISE) {
    // Roy et al. (2016) RMA, ETM+ = (OLI - intercept) / slope; NIR and SWIR2 only
    sr = sr.subtract(ee.Image.constant([-0.0021, 0.0029])).divide(ee.Image.constant([1.0073, 0.9949]))
           .rename(['nir', 'swir2']);
  }
  return ee.Image(sr.normalizedDifference(['nir', 'swir2']).rename('NBR').updateMask(clear)
           .copyProperties(img, ['system:time_start']));
}

function landsat(id, nir, swir, isOli) {
  return ee.ImageCollection(id).filterBounds(REGION)
    .filterDate(START_YEAR + '-01-01', (CUTOFF + 1) + '-01-01')
    .filter(ee.Filter.calendarRange(MONTHS[0], MONTHS[1], 'month'))
    .map(function (img) { return nbrImage(img, nir, swir, isOli); });
}

var all = landsat('LANDSAT/LT04/C02/T1_L2', 'SR_B4', 'SR_B7', false)
  .merge(landsat('LANDSAT/LT05/C02/T1_L2', 'SR_B4', 'SR_B7', false))
  .merge(landsat('LANDSAT/LE07/C02/T1_L2', 'SR_B4', 'SR_B7', false))
  .merge(landsat('LANDSAT/LC08/C02/T1_L2', 'SR_B5', 'SR_B7', true));

// ---- One NBR value per year: median of all clear observations ----------------
var years = ee.List.sequence(START_YEAR, CUTOFF);
var EMPTY = ee.Image.constant(0).toFloat().rename('NBR').updateMask(0);
var annual = ee.ImageCollection.fromImages(years.map(function (y) {
  y = ee.Number(y);
  var c = all.filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y.add(1), 1, 1));
  var img = ee.Image(ee.Algorithms.If(c.size().gt(0), c.median(), EMPTY));
  return img.toFloat().rename('NBR').set('system:time_start', ee.Date.fromYMD(y, 7, 1).millis());
}));
var nbrStack = annual.toBands().rename(YEAR_NAMES);         // nbr_1985 ... nbr_CUTOFF

// ---- LandTrendr on NBR x 1000, flipped so that a cut is a positive change ------
var ltInput = annual.map(function (img) {
  return ee.Image(img.multiply(-1000).toFloat().copyProperties(img, ['system:time_start']));
});
var params = JSON.parse(JSON.stringify(LT_PARAMS));
params.timeSeries = ltInput;
var lt = ee.Algorithms.TemporalSegmentation.LandTrendr(params);

// ---- small array helpers (all arrays below are 1 x n rows) -------------------
function one(v) { return ee.Image.constant(v).toFloat().toArray().toArray(1); }            // 1 x 1
function constRow(v, n) {                                                                  // 1 x n
  var vals = []; for (var k = 0; k < n; k++) { vals.push(v); }
  return ee.Image.constant(vals).toFloat().toArray().toArray(1).arrayTranspose();
}
function nextSeg(a, pad) { return a.toFloat().arraySlice(1, 1).arrayCat(one(pad), 1); }   // a[k+1]
function arrSum(a) { return a.arrayReduce(ee.Reducer.sum(), [1]).arrayGet([0, 0]); }
function arrMaxPad(a, keep, pad) {   // max of a where keep, or pad if none
  return a.toFloat().arrayMask(keep).arrayCat(one(pad), 1).arrayReduce(ee.Reducer.max(), [1]).arrayGet([0, 0]);
}

// ---- Segments from the LandTrendr vertices ------------------------------------
var ltA = lt.select('LandTrendr');                  // 4 x nYears: year, source, fitted, isVertex
var vtx = ltA.arrayMask(ltA.arraySlice(0, 3, 4));   // 4 x nVertices
var vy  = vtx.arraySlice(0, 0, 1);                  // vertex years
var vf  = vtx.arraySlice(0, 2, 3).multiply(-0.001); // fitted NBR at vertices (back to normal NBR)
// pad with a flat dummy segment so that slices below are never empty
vy = vy.arrayCat(vy.arraySlice(1, -1).add(1), 1);
vf = vf.arrayCat(vf.arraySlice(1, -1), 1);
var y0 = vy.arraySlice(1, 0, -1), y1 = vy.arraySlice(1, 1);
var f0 = vf.arraySlice(1, 0, -1), f1 = vf.arraySlice(1, 1);
var dur = y1.subtract(y0).toFloat();
var dv  = f1.subtract(f0).toFloat();                // NBR change per segment (negative = loss)

var isCut = dv.lte(-DROP).and(dur.lte(MAX_CUT_YEARS));
// Regrowth after cut k, over the next one or two segments (LandTrendr often splits regrowth into a
// fast and a slow part). Time to regain a target share of the drop, interpolated along the segments.
var drop = dv.multiply(-1);
var r1 = nextSeg(dv, 0),  d1 = nextSeg(dur, 99);
var r2 = nextSeg(r1, 0),  d2 = nextSeg(d1, 99);
function timeTo(frac) {
  var tgt = drop.multiply(frac);
  var in1 = r1.gte(tgt);
  var t1 = d1.multiply(tgt.divide(r1.max(1e-6)));
  var in2 = in1.not().and(r1.gt(0)).and(r2.gt(0)).and(r1.add(r2).gte(tgt));
  var t2 = d1.add(d2.multiply(tgt.subtract(r1).divide(r2.max(1e-6))));
  return t1.multiply(in1).add(t2.multiply(in2)).add(in1.not().and(in2.not()).multiply(999));
}
var t50 = timeTo(REC_FRAC);
var isCycle = isCut.and(t50.lte(MAX_REC_YEARS));
var nCuts = arrSum(isCut).rename('n_cuts');

// ---- Yearly-value features (same definitions as the offline analysis) --------
var yrsRow = ee.Image.constant(YEARS_JS).toFloat().toArray().toArray(1).arrayTranspose();   // 1 x NY years
var yv = nbrStack.unmask(-9).toArray().toArray(1).arrayTranspose();   // 1 x NY yearly NBR, -9 = no data
var valid = yv.gt(-5);
var yearsBelow = arrSum(yv.lt(FOREST_LEVEL).and(valid));
var low  = yv.lte(OPEN_LEVEL).and(valid).toFloat();
var high = yv.gte(FOREST_LEVEL).and(valid);
function shiftRight(a, k, pad) { return constRow(pad, k).arrayCat(a.arraySlice(1, 0, NY - k), 1); }
var lowCum  = low.arrayAccum(1, ee.Reducer.sum());
var lowWin  = shiftRight(lowCum, 1, 0).subtract(shiftRight(lowCum, 9, 0));            // open years in t-8..t-1
var lastLow = shiftRight(yrsRow.multiply(low).arrayAccum(1, ee.Reducer.max()), 1, -99); // last open year < t
var isStart = high.and(lowWin.gte(MIN_OPEN_YEARS)).and(yrsRow.subtract(lastLow).lte(FAST_RISE_YEARS));

var feats = ee.Image.cat(
  nCuts,
  arrSum(isCycle).rename('n_cycles'),
  arrMaxPad(y0.add(1), isCut, 0).rename('last_cut_year'),                // 0 = no cut
  arrMaxPad(drop, isCut, 0).rename('max_drop'),
  nCuts.eq(0).and(yearsBelow.lte(STABLE_TOLERANCE)).rename('forest_since_1985'),
  isStart.arrayReduce(ee.Reducer.max(), [1]).arrayGet([0, 0]).rename('abrupt_start'),
  yearsBelow.rename('years_below_forest'),
  vy.arrayLength(1).subtract(1).rename('n_vertices')
).toFloat();

var ly = hansen.select('lossyear');
var hansenBands = ee.Image.cat(
  hansen.select('treecover2000').rename('hansen_tc2000'),
  hansen.select('gain').rename('hansen_gain'),
  ly.rename('hansen_lossyear'),
  ly.gte(1).and(ly.lte(CUTOFF - 2000)).rename('hansen_loss_le_cutoff')
).toFloat();

var out = feats.addBands(nbrStack).addBands(hansenBands);

// =============================================================================
if (MODE === 'CALIBRATE') {
  // Closed canopy using information up to the cutoff only; tree cover is known from 2000 on.
  var closed = hansen.select('treecover2000').gte(90)
    .and(ly.eq(0).or(ly.gt(CUTOFF - 2000))).and(hansen.select('gain').eq(0));
  var calYears = YEAR_NAMES.filter(function (n) { return Number(n.slice(4)) >= 2000; });
  var samp = nbrStack.select(calYears).updateMask(closed).sample({
    region: states.geometry(), scale: 30, numPixels: 20000, seed: 7, tileScale: 16, geometries: false});
  // pool all yearly values of all sampled pixels, then take percentiles (same statistic as offline)
  var flat = ee.FeatureCollection(calYears.map(function (b) {
    return samp.select([b], ['v']);
  })).flatten();
  var st = flat.reduceColumns(ee.Reducer.percentile([1, 5, 50]), ['v']);
  // open land: <= 10% tree cover in 2000, no gain; yearly values 1999-2001
  var openLand = hansen.select('treecover2000').lte(10).and(hansen.select('gain').eq(0));
  var openYears = ['nbr_1999', 'nbr_2000', 'nbr_2001'];
  var sampO = nbrStack.select(openYears).updateMask(openLand).sample({
    region: states.geometry(), scale: 30, numPixels: 20000, seed: 11, tileScale: 16, geometries: false});
  var flatO = ee.FeatureCollection(openYears.map(function (b) { return sampO.select([b], ['v']); })).flatten();
  var stO = flatO.reduceColumns(ee.Reducer.percentile([25, 50]), ['v']);
  var calib = ee.FeatureCollection([ee.Feature(null, {
    closed_p1: st.get('p1'), closed_p5: st.get('p5'), closed_p50: st.get('p50'),
    open_p25: stO.get('p25'), open_p50: stO.get('p50'),
    n_closed_pixels: samp.size(), n_open_pixels: sampO.size(), drop: DROP})]);
  Export.table.toDrive({collection: calib, description: 'harvest_calibration_' + CUTOFF, fileFormat: 'CSV'});
  print('Run the export task harvest_calibration_' + CUTOFF + ' (Tasks tab). Then set FOREST_LEVEL = closed_p5 ' +
        'and OPEN_LEVEL = min(open_p25, closed_p5 - DROP / 2). (An interactive print would likely time out.)');

} else if (MODE === 'MAP') {
  Map.centerObject(ee.Geometry.Point([-50.0, -26.5]), 11);
  Map.addLayer(nbrStack.select('nbr_' + CUTOFF), {min: 0, max: 0.9, palette: ['8c510a', 'f6e8c3', '01665e']},
               'NBR ' + CUTOFF, false);
  Map.addLayer(feats.select('n_cuts').selfMask(), {min: 1, max: 4, palette: ['fee08b', 'f46d43', 'a50026']},
               'Cuts before ' + (CUTOFF + 1));
  Map.addLayer(feats.select('forest_since_1985').selfMask(), {palette: ['1a9850']}, 'Forest since 1985', false);
  Map.addLayer(feats.select('abrupt_start').selfMask(), {palette: ['7b3294']}, 'Abrupt start', false);
  Map.addLayer(qc.style({color: '000000', fillColor: '00000000', width: 2}), {}, 'QC polygons', false);
  print('Click a pixel with the Inspector to see its features and yearly NBR.');

} else {
  // ---- EXPORT: validation points (same filter as the team's 2022 analysis) ------
  var val = val3.filter(ee.Filter.eq('BORDA_2022', 0)).filter(ee.Filter.inList('CLASS_2022', [3, 9]));
  var keepVal = ['TARGETID', 'CD_MUN', 'CLASS_2020', 'CLASS_2022'];
  var valOut = out.reduceRegions({collection: val.select(keepVal), reducer: ee.Reducer.first(),
                                  scale: 30, tileScale: 16});
  Export.table.toDrive({collection: valOut, description: 'harvest_features_val_' + CUTOFF,
                        fileFormat: 'CSV', selectors: keepVal.concat(OUT_NAMES)});
  // ---- EXPORT: QC polygons, one pixel inside each polygon ---------------------
  // Centroid of the largest part: the plain centroid of a MultiPolygon can fall between its parts (outside
  // the clearing for ~21% of QC features). Parts are small pixel-shaped blocks, so their centroid is inside
  // or, for 11 non-convex parts, at most ~4 m outside - far below the 30 m pixel.
  // The offline analysis used shapely's representative_point(); to sample exactly the same pixels,
  // upload the QC rows of CROPIN-MIT-DATA/harvest_cycles/points.csv as a point asset and use it here.
  var qcPts = qc.map(function (f) {
    var parts = f.geometry().geometries();
    var areas = parts.map(function (g) { return ee.Geometry(g).area(1); });
    var big = ee.Geometry(parts.get(areas.indexOf(areas.reduce(ee.Reducer.max()))));
    return ee.Feature(big.centroid(1), {qc_label: f.get('qc_label'), qc_id: f.id()});
  });
  var qcOut = out.reduceRegions({collection: qcPts, reducer: ee.Reducer.first(), scale: 30, tileScale: 16});
  Export.table.toDrive({collection: qcOut, description: 'harvest_features_qc_' + CUTOFF,
                        fileFormat: 'CSV', selectors: ['qc_id', 'qc_label'].concat(OUT_NAMES)});
  print('Open the Tasks tab and run both exports.');
}
