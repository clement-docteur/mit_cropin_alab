// #####################################################################
// 06_map_view_winter_check.js  (Meera, Nov 2026)
// WHAT IT DOES
//   Deepak (11/12 meeting) was surprised that winter imagery helps: in Cropin's experience winter
//   numbers looked good on paper but the map didn't. This script draws the notebook-04 result as a
//   map over a small test area, so we can check it by eye:
//     - the raw season layers (summer SAG, winter SAG, winter NDMI) on top of the satellite basemap;
//     - plantation probability from two models: SAG Oct-Jan only, and the 6-feature short list;
//     - where the two models disagree (the pixels the winter measurements change);
//     - MapBiomas 10 m 2020 plantation and forest, for comparison;
//     - 75 random tree pixels to label blind: 25 where only B says plantation, 25 where only A does,
//       and 25 where the two models agree (stratifiedSample also draws the unlisted class 0).
//       The agreement points are a control: how good are both models where they agree?
//   The models are refit by src/map_model_weights.py on reference points OUTSIDE the test areas,
//   so the map is drawn by a model that never saw the points inside them.
// ASSETS NEEDED (GEE project cropin-510901)
//   projects/cropin-510901/assets/mapbiomas_ref_south  (reference points, shown on request)
//   projects/cropin-510901/assets/cropin_qc            (QC polygons, shown on request)
//   Public: Sentinel-2 SR harmonized, Cloud Score+, Meta canopy height, MapBiomas 10 m C2.
// HOW TO RUN
//   1. Paste into a new script in the Code Editor (project cropin-510901). Set AREA below. Run.
//   2. Use the Layers menu (top right of the map) to toggle layers. Layers start switched off
//      where they would give away the answer before you look (reference points, QC polygons).
//   3. For the blind check, pick a point number in the panel on the left; the map zooms to it.
//   4. Tasks tab: RUN 'map_check_key_<AREA>' only AFTER labelling. It holds which model said what.
// SAME CHOICES AS gee/05 AND NOTEBOOK 04
//   2019-2022, CLOUDY_PIXEL_PERCENTAGE < 90, Cloud Score+ cs >= 0.6, monthly medians, seasons are the
//   average of the monthly values (Oct-Jan, Apr-Sep), SAG with Cropin's dark-pixel rule, NDVI and
//   NDMI from the monthly band medians. Each pixel is averaged over a 15 m circle, like a reference
//   point. Trees only: Meta canopy height >= 5 m (Deepak's suggestion).
// #####################################################################

// <<WEIGHTS (written by src/map_model_weights.py)
var CONFIG = {
  "areas": [
    {
      "name": "PR",
      "west": -50.5,
      "south": -24.4,
      "east": -50.4,
      "north": -24.3,
      "n_plantation_points": 3,
      "n_forest_points": 2
    },
    {
      "name": "RS",
      "west": -50.2,
      "south": -28.8,
      "east": -50.1,
      "north": -28.7,
      "n_plantation_points": 4,
      "n_forest_points": 1
    },
    {
      "name": "SC",
      "west": -49.2,
      "south": -28.0,
      "east": -49.1,
      "north": -27.9,
      "n_plantation_points": 2,
      "n_forest_points": 4
    }
  ],
  "models": {
    "sag": {
      "intercept": -57.98529286,
      "weights": {
        "SAG_octjan": 0.36458596
      },
      "cutoff": 0.558
    },
    "six": {
      "intercept": -65.93589566,
      "weights": {
        "SAG_octjan": 0.29642872,
        "B6_winter": 0.00691004,
        "B8A_winter": -0.0034514,
        "SAG_winter": 0.15384696,
        "NDMI_winter": 14.31975275,
        "NDVI_winter": -18.39324828
      },
      "cutoff": 0.4677
    }
  }
};
// WEIGHTS>>

// ---------------- settings you can change ----------------
var AREA = 'PR';                               // 'PR', 'SC' or 'RS'
var START = '2019-01-01', END = '2023-01-01';  // try '2021-01-01'/'2022-01-01' to test a single year
var N_PER_SIDE = 25;                           // blind points per group: only B, only A, both agree (75 in all)
var MIN_TREE_M = 5;                            // Meta canopy height threshold for "tree"

// ---------------- area ----------------
var PROJECT = 'cropin-510901';
var area = CONFIG.areas.filter(function (a) { return a.name === AREA; })[0];
var region = ee.Geometry.Rectangle([area.west, area.south, area.east, area.north]);
Map.setOptions('SATELLITE');
Map.centerObject(region, 13);
Map.addLayer(ee.Image().paint(region, 0, 2), {palette: ['ffffff']}, 'Test area outline');

// ---------------- Sentinel-2, same as gee/05 ----------------
var OCTJAN = [10, 11, 12, 1], WINTER = [4, 5, 6, 7, 8, 9];
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region).filterDate(START, END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
  .linkCollection(csPlus, csPlus.first().bandNames())
  .map(function (img) { return img.updateMask(img.select('cs').gte(0.6)); });

function sagOf(img) {                       // identical to gee/03-05 (Cropin's addGreenAngle())
  var yB = img.select('B2').divide(10000), yG = img.select('B3').divide(10000), yR = img.select('B4').divide(10000);
  var xB = 0.4924, xG = 0.5598, xR = 0.6646;
  var dx1 = xB - xG, dy1 = yB.subtract(yG), dx2 = xR - xG, dy2 = yR.subtract(yG);
  var dot = dy1.multiply(dy2).add(dx1 * dx2);
  var m1 = dy1.pow(2).add(dx1 * dx1).sqrt(), m2 = dy2.pow(2).add(dx2 * dx2).sqrt();
  var sag = dot.divide(m1.multiply(m2)).clamp(-1, 1).acos().multiply(180 / Math.PI).rename('SAG');
  return sag.updateMask(yG.lte(0.05).and(yR.lte(0.05)).and(yB.lte(0.05)));
}

function monthImage(m) {                    // one calendar month, pooled over the years
  var col = s2.filter(ee.Filter.calendarRange(m, m, 'month'));
  var med = col.select(['B4', 'B6', 'B8', 'B8A', 'B11']).median();
  var ndvi = med.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndmi = med.normalizedDifference(['B8', 'B11']).rename('NDMI');
  return med.select(['B6', 'B8A']).addBands([ndvi, ndmi, col.map(sagOf).median(),
                                              col.select('B4').count().rename('n')]);
}
function season(months) {                   // average of the monthly values; masked months are skipped
  return ee.ImageCollection(months.map(monthImage)).mean();
}
var summer = season(OCTJAN), winter = season(WINTER);
var feats = ee.Image.cat([
  summer.select('SAG').rename('SAG_octjan'),
  winter.select('B6').rename('B6_winter'), winter.select('B8A').rename('B8A_winter'),
  winter.select('SAG').rename('SAG_winter'), winter.select('NDMI').rename('NDMI_winter'),
  winter.select('NDVI').rename('NDVI_winter')
]).focalMean({radius: 15, kernelType: 'circle', units: 'meters'})   // like a 15 m reference buffer
  .clip(region);

var meta = ee.ImageCollection('projects/sat-io/open-datasets/facebook/meta-canopy-height').mosaic();
var trees = meta.gte(MIN_TREE_M);
feats = feats.updateMask(trees);

// ---------------- the two models ----------------
function prob(key) {
  var m = CONFIG.models[key], cols = Object.keys(m.weights);
  var logit = feats.select(cols).multiply(ee.Image.constant(cols.map(function (c) { return m.weights[c]; })))
    .reduce(ee.Reducer.sum()).add(m.intercept);
  return ee.Image(1).divide(logit.multiply(-1).exp().add(1)).rename('p_' + key);
}
var pSag = prob('sag'), pSix = prob('six');
var flagSag = pSag.gte(CONFIG.models.sag.cutoff), flagSix = pSix.gte(CONFIG.models.six.cutoff);
// 1 = only the 6-feature model calls it plantation ("winter adds"), 2 = only SAG does ("winter removes")
var dis = flagSix.and(flagSag.not()).multiply(1).add(flagSag.and(flagSix.not()).multiply(2))
  .rename('dis').updateMask(pSag.mask().and(pSix.mask()));

// ---------------- layers ----------------
var PROB = {min: 0, max: 1, palette: ['3B6EA8', 'f7f7f7', '750014']};    // blue forest → burgundy plantation
var mb10 = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1')
  .select('classification_2020').clip(region);
Map.addLayer(trees.selfMask().clip(region), {palette: ['00ff00']}, 'Trees (Meta height >= ' + MIN_TREE_M + ' m)', false, 0.3);
Map.addLayer(mb10.eq(3).selfMask(), {palette: ['3B6EA8']}, 'MapBiomas 10 m 2020: forest', false, 0.6);
Map.addLayer(mb10.eq(9).selfMask(), {palette: ['750014']}, 'MapBiomas 10 m 2020: plantation', false, 0.6);

// raw season layers, stretched to the 2nd-98th percentile inside the area
var RAW = ['SAG_octjan', 'SAG_winter', 'NDMI_winter'];
feats.select(RAW).reduceRegion({reducer: ee.Reducer.percentile([2, 98]), geometry: region, scale: 30,
                                maxPixels: 1e9, bestEffort: true})
  .evaluate(function (pc) {
    RAW.forEach(function (b) {
      Map.addLayer(feats.select(b), {min: pc[b + '_p2'], max: pc[b + '_p98'], palette: ['000000', 'ffffff']},
                   'Raw: ' + b + ' (bright = high)', false);
    });
  });
Map.addLayer(winter.select('n').clip(region), {min: 0, max: 40, palette: ['ff9900', 'ffffff']},
             'Clear winter images per pixel (orange = few)', false);
Map.addLayer(pSag, PROB, 'Model A: SAG Oct-Jan only (probability)', false);
Map.addLayer(pSix, PROB, 'Model B: 6 features (probability)', false);
Map.addLayer(flagSag.selfMask().updateMask(pSag.mask()), {palette: ['750014']}, 'Model A flags plantation (<= 5% forest hidden)', false);
Map.addLayer(flagSix.selfMask().updateMask(pSix.mask()), {palette: ['750014']}, 'Model B flags plantation (<= 5% forest hidden)', false);
Map.addLayer(dis.selfMask(), {min: 1, max: 2, palette: ['ffd400', '00e5ff']},
             'Disagreement: yellow = only B (winter adds), cyan = only A (winter removes)', false);

var ref = ee.FeatureCollection('projects/' + PROJECT + '/assets/mapbiomas_ref_south').filterBounds(region);
Map.addLayer(ref.filter(ee.Filter.eq('label', 'plantation')).style({color: '750014', pointSize: 5}),
             {}, 'Reference points: plantation (answer key)', false);
Map.addLayer(ref.filter(ee.Filter.eq('label', 'forest')).style({color: '3B6EA8', pointSize: 5}),
             {}, 'Reference points: forest (answer key)', false);
var qc = ee.FeatureCollection('projects/' + PROJECT + '/assets/cropin_qc').filterBounds(region);
Map.addLayer(qc.style({color: 'ffffff', fillColor: '00000000', width: 2}), {}, 'Cropin QC polygons', false);

// ---------------- share of tree pixels each model flags ----------------
ee.Image.cat([flagSag.rename('A'), flagSix.rename('B'), dis.eq(1).rename('only_B'), dis.eq(2).rename('only_A')])
  .updateMask(dis.mask()).reduceRegion({reducer: ee.Reducer.mean(), geometry: region, scale: 10, maxPixels: 1e9,
                                        bestEffort: true})
  .evaluate(function (r) {
    print('Share of tree pixels flagged as plantation in ' + AREA + ':',
          'A (SAG only) ' + (100 * r.A).toFixed(1) + '%, B (6 features) ' + (100 * r.B).toFixed(1) + '%',
          'only B: ' + (100 * r.only_B).toFixed(1) + '%, only A: ' + (100 * r.only_A).toFixed(1) + '%');
  });

// ---------------- blind check: 75 random tree pixels (25 per group) ----------------
// Kept as first run on PR and SC, so all three areas use the same design.
var sample = dis.addBands([pSag, pSix]).stratifiedSample({
  numPoints: N_PER_SIDE, classBand: 'dis', classValues: [1, 2], classPoints: [N_PER_SIDE, N_PER_SIDE],
  region: region, scale: 10, seed: 0, geometries: true
}).randomColumn('r', 1).sort('r');
var n = sample.size();
var pts = ee.FeatureCollection(ee.List.sequence(0, n.subtract(1)).map(function (i) {
  var f = ee.Feature(sample.toList(n).get(i));
  var c = f.geometry().coordinates();
  return f.set({pid: ee.Number(i).add(1), lon: c.get(0), lat: c.get(1)});
}));
// The map layer carries only the point number, so clicking a point does not reveal the answer.
var blind = pts.map(function (f) { return ee.Feature(f.geometry(), {pid: f.get('pid')}); });
Map.addLayer(blind.style({color: 'ff00ff', pointSize: 4, width: 1}), {}, 'Blind check points (magenta)');

Export.table.toDrive({
  collection: pts, description: 'map_check_key_' + AREA, folder: 'cropin_exports', fileFormat: 'CSV',
  selectors: ['pid', 'dis', 'p_sag', 'p_six', 'lon', 'lat']
});

var panel = ui.Panel({style: {width: '260px'}});
panel.add(ui.Label('Blind check, ' + AREA, {fontWeight: 'bold'}));
panel.add(ui.Label('Pick a point, look at the basemap (zoom in, toggle layers OFF), and write down ' +
                   'plantation / forest / unsure. Do not open the key until all are done.'));
var info = ui.Label('');
blind.aggregate_array('pid').evaluate(function (ids) {
  var sel = ui.Select({
    items: ids.map(function (i) { return String(i); }), placeholder: 'Point number',
    onChange: function (v) {
      var g = blind.filter(ee.Filter.eq('pid', Number(v))).geometry();
      Map.centerObject(g, 18);
      info.setValue('Point ' + v + ' of ' + ids.length);
    }
  });
  panel.add(sel); panel.add(info);
});
ui.root.insert(0, panel);
print('Area ' + AREA + ': ' + area.n_plantation_points + ' plantation and ' + area.n_forest_points +
      ' forest reference points inside (held out of the model). Cutoffs: A ' + CONFIG.models.sag.cutoff +
      ', B ' + CONFIG.models.six.cutoff);
