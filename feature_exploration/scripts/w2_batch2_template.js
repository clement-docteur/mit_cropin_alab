// ============================================================================
// Cropin x MIT – Feature Exploration, Wave 2, Batch 2 (Family B seasonal + Family F radar)
// Plan: Feature Exploration/00 Workflow Plan.md (v4.1). Written by Claude, 2026-11-06.
//
// HOW TO RUN (about 3 minutes of clicking):
//   1. Paste into code.earthengine.google.com, click Run. Console prints "points __N__".
//   2. Tasks tab: click RUN on all 6 tasks (default settings). Expected 10–60 min each.
//   Each task is independent; if one fails, tell Claude which.
//
// TIME WINDOW (as-of rule): for every point, the 3 calendar years ending at its cutoff year
// (MapBiomas: 2020–2022; QC: the 3 years before the estimated clearing year, mostly 2023–2025).
//
// FILES PRODUCED (folder cropin_texture):
//   w2_s2_monthly_mb.csv  B (H21–H23): monthly Sentinel-2 medians (B2 B3 B4 B5 B8 B8A B11) + clear count, MapBiomas
//   w2_s2_monthly_qc.csv  same for QC points
//   w2_s1.csv             F (H25): Sentinel-1 C-band VV/VH: median, p10, p90, SD over time, incidence angle
//   w2_palsar.csv         F (H26): ALOS-2 PALSAR-2 L-band HH/HV yearly mosaics, all available years
//   w2_lst.csv            B (H27): Landsat 8/9 land surface temperature, summer and winter, point and 1 km mean
//   w2_chirps.csv         B (H23): CHIRPS monthly rainfall and the 1991–2020 mean for that month (to define drought months)
// ============================================================================
var PTS = __PTS__;   // [lon, lat, label, id, set, cutoff]
var allPts = ee.FeatureCollection(PTS.map(function(r) {
  return ee.Feature(ee.Geometry.Point([r[0], r[1]]), {label: r[2], id: r[3], set: r[4], cutoff: r[5]});
}));
print('points', allPts.size());
var CUTOFFS = {mb: __CUT_MB__, qc: __CUT_QC__};   // only non-empty (set, cutoff) groups
function win(y) { return [ee.Date.fromYMD(y - 2, 1, 1), ee.Date.fromYMD(y + 1, 1, 1)]; }
function strip(fc) { return fc.map(function(f) { return ee.Feature(null, f.toDictionary()); }); }

// ---------- 1–2. Sentinel-2 monthly medians (Cloud Score+ cs >= 0.6) ----------
var CS = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var S2B = ['B2', 'B3', 'B4', 'B5', 'B8', 'B8A', 'B11'];
function s2monthly(pts, y) {
  var reg = pts.geometry().bounds();
  return ee.FeatureCollection(ee.List.sequence(0, 35).map(function(i) {
    var start = win(y)[0].advance(i, 'month');
    var c = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(reg)
      .filterDate(start, start.advance(1, 'month')).linkCollection(CS, ['cs'])
      .map(function(im) { return im.updateMask(im.select('cs').gte(0.6)); }).select(S2B);
    var empty = ee.Image.constant([0, 0, 0, 0, 0, 0, 0, 0]).rename(S2B.concat(['n_clear'])).updateMask(0).toFloat();
    var img = ee.Image(ee.Algorithms.If(c.size().gt(0),
      c.median().addBands(c.select('B8').count().rename('n_clear')).toFloat(), empty));
    return strip(img.reduceRegions({collection: pts, reducer: ee.Reducer.first(), scale: 10, tileScale: 4}))
      .map(function(f) { return f.set('month', start.format('YYYY-MM')); });
  })).flatten();
}
function byCutoff(setName, fn) {
  return ee.FeatureCollection(CUTOFFS[setName].map(function(y) {
    return fn(allPts.filter(ee.Filter.and(ee.Filter.eq('set', setName), ee.Filter.eq('cutoff', y))), y);
  })).flatten();
}
Export.table.toDrive({collection: byCutoff('mb', s2monthly), description: 'w2_s2_monthly_mb', folder: 'cropin_texture', fileFormat: 'CSV'});
Export.table.toDrive({collection: byCutoff('qc', s2monthly), description: 'w2_s2_monthly_qc', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 3. Sentinel-1 C-band (GRD, IW, VV+VH, values in dB) ----------
function s1stats(pts, y) {
  var w = win(y);
  var c = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(pts.geometry().bounds()).filterDate(w[0], w[1])
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH', 'angle'])
    .map(function(im) { return im.addBands(im.select('VH').subtract(im.select('VV')).rename('VHminusVV')); });
  var red = ee.Reducer.median().combine(ee.Reducer.percentile([10, 90]), '', true)
    .combine(ee.Reducer.stdDev(), '', true).combine(ee.Reducer.count(), '', true);
  var img = c.select(['VV', 'VH', 'VHminusVV']).reduce(red).addBands(c.select('angle').median().rename('angle_median'));
  return strip(img.reduceRegions({collection: pts, reducer: ee.Reducer.first(), scale: 10, tileScale: 8}));
}
Export.table.toDrive({collection: byCutoff('mb', s1stats).merge(byCutoff('qc', s1stats)), description: 'w2_s1',
  folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 4. ALOS-2 PALSAR-2 yearly mosaics (L-band HH/HV, DN; converted to dB in Python) ----------
var palsar = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH').select(['HH', 'HV']);
var palTable = palsar.map(function(im) {
  return strip(im.reduceRegions({collection: allPts, reducer: ee.Reducer.first(), scale: 25, tileScale: 4}))
    .map(function(f) { return f.set('year', ee.Date(im.get('system:time_start')).get('year')); });
}).flatten();
Export.table.toDrive({collection: palTable, description: 'w2_palsar', folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 5. Landsat 8/9 land surface temperature (ST_B10, Kelvin) ----------
function lst(pts, y) {
  var w = win(y);
  function get(id) {
    return ee.ImageCollection(id).filterBounds(pts.geometry().bounds()).filterDate(w[0], w[1]).map(function(im) {
      var qa = im.select('QA_PIXEL');
      var clear = qa.bitwiseAnd(1 << 1).eq(0).and(qa.bitwiseAnd(1 << 3).eq(0)).and(qa.bitwiseAnd(1 << 4).eq(0));
      return im.select('ST_B10').multiply(0.00341802).add(149.0).rename('LST').updateMask(clear)
        .copyProperties(im, ['system:time_start']);
    });
  }
  var c = get('LANDSAT/LC08/C02/T1_L2').merge(get('LANDSAT/LC09/C02/T1_L2'));
  var summer = c.filter(ee.Filter.or(ee.Filter.calendarRange(12, 12, 'month'), ee.Filter.calendarRange(1, 3, 'month'))).median().rename('LST_summer');
  var winter = c.filter(ee.Filter.calendarRange(6, 8, 'month')).median().rename('LST_winter');
  var img = summer.addBands(winter);
  var atPt = strip(img.reduceRegions({collection: pts, reducer: ee.Reducer.first(), scale: 30, tileScale: 4}));
  var around = strip(img.rename(['LST_summer_1km', 'LST_winter_1km'])
    .reduceRegions({collection: pts.map(function(f) { return f.buffer(1000); }), reducer: ee.Reducer.mean(), scale: 30, tileScale: 8}));
  return atPt.merge(around.map(function(f) { return f.set('kind', '1km'); }));
}
Export.table.toDrive({collection: byCutoff('mb', lst).merge(byCutoff('qc', lst)), description: 'w2_lst',
  folder: 'cropin_texture', fileFormat: 'CSV'});

// ---------- 6. CHIRPS monthly rainfall vs 1991–2020 climatology (drought months for H23) ----------
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').select('precipitation');
var clim = ee.ImageCollection(ee.List.sequence(1, 12).map(function(m) {
  var yrs = ee.List.sequence(1991, 2020).map(function(y) {
    var s = ee.Date.fromYMD(y, m, 1); return chirps.filterDate(s, s.advance(1, 'month')).sum();
  });
  return ee.ImageCollection(yrs).mean().rename('clim_mm').set('m', m);
}));
function rain(pts, y) {
  return ee.FeatureCollection(ee.List.sequence(0, 35).map(function(i) {
    var s = win(y)[0].advance(i, 'month');
    var img = chirps.filterDate(s, s.advance(1, 'month')).sum().rename('rain_mm')
      .addBands(ee.Image(clim.filter(ee.Filter.eq('m', s.get('month'))).first()));
    return strip(img.reduceRegions({collection: pts, reducer: ee.Reducer.first(), scale: 5566}))
      .map(function(f) { return f.set('month', s.format('YYYY-MM')); });
  })).flatten();
}
Export.table.toDrive({collection: byCutoff('mb', rain).merge(byCutoff('qc', rain)), description: 'w2_chirps',
  folder: 'cropin_texture', fileFormat: 'CSV'});
