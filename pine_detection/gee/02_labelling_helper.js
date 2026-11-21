// #####################################################################
// pine_detection/gee/02_labelling_helper.js  (Clement, Nov 2026)
// WHAT IT DOES
//   Helps with photo-interpretation of the pine vs eucalyptus gold sample
//   (pine_detection/docs/labelling_protocol.md). Pick a sample_id and it shows:
//     - the point and its 15 m circle on Google's high-resolution basemap;
//     - Sentinel-2 summer (Dec 2021-Feb 2022) and winter (Jun-Aug 2022) composites, in true colour and
//       in false colour (NIR / SWIR-1 / red: pine looks darker than eucalyptus);
//     - the yearly Landsat NBR history 1985-2024 (clear-cuts are sharp drops; eucalyptus about every
//       6-8 years, pine about every 12-25 years with smaller thinning drops);
//     - Sentinel-2 NDVI 2019-2024 at the point.
//   It shows NO SDPT and NO MapBiomas layer, on purpose, so labelling stays blind.
// ASSET NEEDED (GEE project cropin-510901)
//   projects/cropin-510901/assets/species_gold_sample
//     Upload data/processed/gold_sample/gold_points_gee.csv as a table asset (Assets > NEW > CSV),
//     with X = lon and Y = lat. Columns: sample_id, batch, lon, lat (public MapBiomas points only).
// HOW TO RUN
//   Paste into a new script in the Code Editor (project cropin-510901) and press Run. Type a sample_id
//   (for example G001) or use Previous / Next. Write your labels in your own gold_sheet CSV, not here.
// CHOICES
//   - Same Sentinel-2 cloud handling as gee/03-05: CLOUDY_PIXEL_PERCENTAGE < 90, Cloud Score+ cs >= 0.6.
//   - Landsat Collection 2 Level 2, scaled to reflectance, cloud/shadow masked with QA_PIXEL.
//     NBR = (NIR - SWIR2) / (NIR + SWIR2). Yearly median of clear observations at the point.
// #####################################################################

var PROJECT = 'cropin-510901';
var pts = ee.FeatureCollection('projects/' + PROJECT + '/assets/species_gold_sample');
var BUFFER_M = 15;

// ---------------------------------------------------------------- Sentinel-2
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
function s2(start, end) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
    .linkCollection(csPlus, ['cs'])
    .map(function (img) {
      var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
      return img.updateMask(img.select('cs').gte(0.6)).addBands(ndvi);
    });
}
var summer = s2('2021-12-01', '2022-03-01').median();
var winter = s2('2022-06-01', '2022-09-01').median();
var TRUE_VIS = {bands: ['B4', 'B3', 'B2'], min: 0, max: 1500};
var FALSE_VIS = {bands: ['B8', 'B11', 'B4'], min: 0, max: 4500};

// ---------------------------------------------------------------- Landsat NBR, yearly
function prepLandsat(nir, swir2) {
  return function (img) {
    var qa = img.select('QA_PIXEL');
    var clear = qa.bitwiseAnd(1 << 1).eq(0)        // dilated cloud
      .and(qa.bitwiseAnd(1 << 3).eq(0))            // cloud
      .and(qa.bitwiseAnd(1 << 4).eq(0));           // cloud shadow
    var sr = img.select([nir, swir2]).multiply(0.0000275).add(-0.2);
    return sr.normalizedDifference([nir, swir2]).rename('NBR')
      .updateMask(clear).copyProperties(img, ['system:time_start']);
  };
}
var landsat = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2').map(prepLandsat('SR_B4', 'SR_B7'))
  .merge(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2').map(prepLandsat('SR_B4', 'SR_B7')))
  .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').map(prepLandsat('SR_B5', 'SR_B7')))
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').map(prepLandsat('SR_B5', 'SR_B7')));

function yearlyNBR(region) {
  var years = ee.List.sequence(1985, 2024);
  return ee.ImageCollection.fromImages(years.map(function (y) {
    var start = ee.Date.fromYMD(y, 1, 1);
    return landsat.filterBounds(region).filterDate(start, start.advance(1, 'year')).median()
      .set('system:time_start', start.advance(6, 'month').millis());
  })).filter(ee.Filter.listContains('system:band_names', 'NBR'));
}

// ---------------------------------------------------------------- UI
Map.setOptions('SATELLITE');
var panel = ui.Panel({style: {width: '430px'}});
ui.root.insert(0, panel);
panel.add(ui.Label('Species gold sample: labelling helper',
                   {fontWeight: 'bold', fontSize: '16px'}));
panel.add(ui.Label('Blind view: no SDPT or MapBiomas layers. Record labels in your own CSV sheet.',
                   {fontSize: '11px', color: '#666'}));
var idBox = ui.Textbox({placeholder: 'sample_id, e.g. G001'});
var status = ui.Label('');
var charts = ui.Panel();
var ids = [];
var current = -1;

function show(i) {
  if (i < 0 || i >= ids.length) { return; }
  current = i;
  var sid = ids[i];
  idBox.setValue(sid, false);
  var f = ee.Feature(pts.filter(ee.Filter.eq('sample_id', sid)).first());
  var pt = f.geometry();
  var circle = pt.buffer(BUFFER_M);
  Map.layers().reset();
  Map.addLayer(summer, TRUE_VIS, 'S2 summer 2021-22, true colour', false);
  Map.addLayer(winter, TRUE_VIS, 'S2 winter 2022, true colour', false);
  Map.addLayer(summer, FALSE_VIS, 'S2 summer, NIR/SWIR1/red', false);
  Map.addLayer(winter, FALSE_VIS, 'S2 winter, NIR/SWIR1/red', false);
  Map.addLayer(ee.Image().paint(ee.FeatureCollection([ee.Feature(circle)]), 0, 2),
               {palette: ['FFFF00']}, sid + ' (15 m circle)');
  Map.centerObject(pt, 17);
  status.setValue(sid + '  (' + (i + 1) + ' of ' + ids.length + ')');

  charts.clear();
  charts.add(ui.Chart.image.series(yearlyNBR(pt), pt, ee.Reducer.mean(), 30)
    .setOptions({title: 'Landsat NBR, yearly median (1985-2024)', legend: {position: 'none'},
                 vAxis: {title: 'NBR', viewWindow: {min: -0.2, max: 1}}, hAxis: {title: 'year'},
                 pointSize: 3, lineWidth: 1, colors: ['#2F5D50']}));
  charts.add(ui.Chart.image.series(s2('2019-01-01', '2025-01-01').select('NDVI'), circle,
                                   ee.Reducer.mean(), 10)
    .setOptions({title: 'Sentinel-2 NDVI at the point (2019-2024)', legend: {position: 'none'},
                 vAxis: {title: 'NDVI', viewWindow: {min: 0, max: 1}}, hAxis: {title: 'date'},
                 pointSize: 2, lineWidth: 0, colors: ['#C08A2B']}));
}

idBox.onChange(function (v) {
  var i = ids.indexOf(v.trim().toUpperCase());
  if (i >= 0) { show(i); } else { status.setValue('Unknown sample_id: ' + v); }
});
var nav = ui.Panel([
  ui.Button('Previous', function () { show(current - 1); }),
  ui.Button('Next', function () { show(current + 1); })
], ui.Panel.Layout.flow('horizontal'));
panel.add(idBox);
panel.add(nav);
panel.add(status);
panel.add(ui.Label('Tip: the historical-imagery slider in Google Earth Pro gives the 2012-15 view ' +
                   '(species_2014 column).', {fontSize: '11px', color: '#666'}));
panel.add(charts);

status.setValue('Loading sample ids...');
pts.aggregate_array('sample_id').sort().evaluate(function (list) {
  ids = list;
  show(0);
});
