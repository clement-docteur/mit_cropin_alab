// =============================================================================
// Side-by-side colour check: one time window vs another (swipe to compare)
// Points are coloured by group: orange = missed plantation, green = real forest,
// dark brown = caught plantation. Zoom in (~level 15-16) to see stand colour.
// =============================================================================

// ---- SETTINGS: pick any two windows (start inclusive, end exclusive) ---------
var WIN_A = ['2022-08-01', '2022-11-01'];   // left map:  late winter (Aug-Oct)
var WIN_B = ['2022-12-01', '2023-03-01'];   // right map: summer (Dec-Feb)
var LABEL_A = 'Late winter (Aug-Oct 2022)', LABEL_B = 'Summer (Dec 2022-Feb 2023)';
var groupsAsset = 'projects/cropin-510901/assets/val_monthly_2022';  // has a 'group' column
// -----------------------------------------------------------------------------

var region = ee.Geometry.Rectangle([-58, -34, -47.5, -22]);   // southern Brazil
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

// Cloud-masked median composite for a window (Cloud Score+: keep clear pixels)
function composite(win) {
  return s2.filterBounds(region).filterDate(win[0], win[1])
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .linkCollection(csPlus, ['cs_cdf'])
    .map(function (img) { return img.updateMask(img.select('cs_cdf').gte(0.6)); })
    .median();
}
var compA = composite(WIN_A), compB = composite(WIN_B);

var trueVis  = {bands: ['B4', 'B3', 'B2'], min: 0, max: 1500};    // what the eye sees
var falseVis = {bands: ['B8', 'B11', 'B4'], min: 0, max: 4000};   // NIR-SWIR-red: separates canopy types better

// Points coloured by group
var res = ee.FeatureCollection(groupsAsset);
var GROUPS  = ['missed plantation', 'real forest', 'caught plantation'];
var GCOLORS = ['d68a00', '1f8d49', '7a5900'];

function fillMap(map, comp, label) {
  map.addLayer(comp, trueVis, 'True colour');
  map.addLayer(comp, falseVis, 'False colour (NIR-SWIR-red)', false);
  GROUPS.forEach(function (g, i) {
    map.addLayer(res.filter(ee.Filter.eq('group', g))
                    .style({color: GCOLORS[i], pointSize: 5, fillColor: '00000000', width: 2}),
                 {}, 'Points: ' + g);
  });
  map.add(ui.Label(label, {position: 'top-center', fontWeight: 'bold'}));
}

var left = ui.Map(), right = ui.Map();
fillMap(left, compA, LABEL_A);
fillMap(right, compB, LABEL_B);
ui.Map.Linker([left, right]);   // pan/zoom together

var legend = ui.Panel({style: {position: 'bottom-left', padding: '6px'}});
GROUPS.forEach(function (g, i) {
  legend.add(ui.Panel([
    ui.Label('', {backgroundColor: '#' + GCOLORS[i], padding: '7px', margin: '2px 6px 2px 0'}),
    ui.Label(g, {margin: '2px 0'})
  ], ui.Panel.Layout.Flow('horizontal')));
});
left.add(legend);

left.setOptions('SATELLITE'); right.setOptions('SATELLITE');   // high-res background on both

// Click either map -> yellow ring at the same spot on BOTH maps
var markL = null, markR = null;
function mark(c) {
  var ring = ee.Geometry.Point([c.lon, c.lat]).buffer(25);
  if (markL) { left.layers().remove(markL); right.layers().remove(markR); }
  markL = ui.Map.Layer(ring, {color: 'ffff00'}, 'Clicked spot');
  markR = ui.Map.Layer(ring, {color: 'ffff00'}, 'Clicked spot');
  left.layers().add(markL); right.layers().add(markR);
}
left.onClick(mark); right.onClick(mark);
left.style().set('cursor', 'crosshair'); right.style().set('cursor', 'crosshair');

// Side by side (no swipe): left = WIN_A, right = WIN_B, panned/zoomed together
ui.root.widgets().reset([ui.SplitPanel({firstPanel: left, secondPanel: right, wipe: false})]);
left.setCenter(-51.5, -27.5, 7);
right.setCenter(-51.5, -27.5, 7);

left.setOptions('SATELLITE'); right.setOptions('SATELLITE');