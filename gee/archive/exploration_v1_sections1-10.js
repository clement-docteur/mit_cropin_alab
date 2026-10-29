// ARCHIVED (superseded): first draft of 01 with sections 1–10 only and the QC/ref assets commented out; replaced by 01_exploration_and_benchmark.js.

// =====================================================================
// Cropin – forest vs. plantation: first exploration (southern Brazil)
// Paste into the Google Earth Engine Code Editor and press Run.
// Then CLICK anywhere on the map to get that pixel's NDVI history,
// MapBiomas class, and canopy height in the Console.
// =====================================================================

// ---------- 1. Study area: Paraná, Santa Catarina, Rio Grande do Sul ----------
var states = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Brazil'));
// If the filter below returns nothing, check the exact spellings printed here:
print('Brazil state names:', states.aggregate_array('ADM1_NAME'));
var south = states.filter(ee.Filter.inList('ADM1_NAME',
  ['Parana', 'Santa Catarina', 'Rio Grande Do Sul']));
Map.centerObject(south, 6);

// ---------- 2. MapBiomas land cover (Collection 9, 30m, 1985–2023) ----------
// Class codes: 3 = Forest Formation, 9 = Forest Plantation.
// (The 10m Sentinel-based MapBiomas map Deepak mentioned is a separate asset –
//  ask Cropin for its ID or the script they use, and swap it in later.)
var mb = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1');
var lc2020 = mb.select('classification_2020').clip(south);
var forest2020 = lc2020.eq(3);
var plant2020 = lc2020.eq(9);

// "Stable" labels: same class every year 2016–2023 → more trustworthy training labels
var years = ee.List.sequence(2016, 2023);
var bandNames = years.map(function (y) {
  return ee.String('classification_').cat(ee.Number(y).int().format());
});
var stack = mb.select(bandNames).clip(south);
var stableForest = stack.eq(3).reduce(ee.Reducer.min());
var stablePlant = stack.eq(9).reduce(ee.Reducer.min());

// ---------- 3. Canopy height (ETH global 10m, 2020) ----------
var height = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1').clip(south);

// ---------- 4. AlphaEarth embeddings (64 numbers per 10m pixel per year) ----------
var emb2020 = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate('2020-01-01', '2021-01-01').filterBounds(south).mosaic().clip(south);

// ---------- 5. Sentinel-2 with Cloud Score+ cloud masking ----------
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
function maskClouds(img) {
  return img.updateMask(img.select('cs').gte(0.6));
}
var s2clean = s2.filterBounds(south)
  .linkCollection(csPlus, ['cs'])
  .map(maskClouds);

// True-colour composites for two seasons → eyeball the Dec–Jan "dark green" signal
var summer = s2clean.filterDate('2020-12-01', '2021-02-01').median().clip(south);
var winter = s2clean.filterDate('2020-06-01', '2020-08-01').median().clip(south);

// NDVI time series collection (used by the click tool)
var ndvi = s2clean.filterDate('2019-01-01', '2025-01-01').map(function (img) {
  return img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    .copyProperties(img, ['system:time_start']);
});

// ---------- 6. Map layers (toggle them in the Layers panel) ----------
var rgb = {bands: ['B4', 'B3', 'B2'], min: 0, max: 1500};
Map.addLayer(summer, rgb, 'S2 Dec–Jan 2020/21 (true colour)', false);
Map.addLayer(winter, rgb, 'S2 Jun–Jul 2020 (true colour)', false);
Map.addLayer(emb2020, {bands: ['A01', 'A16', 'A09'], min: -0.3, max: 0.3}, 'AlphaEarth 2020 (3 of 64 dims)', false);
Map.addLayer(height, {min: 0, max: 35, palette: ['ffffcc', '78c679', '006837']}, 'Canopy height 2020 (m)', false);
Map.addLayer(forest2020.selfMask(), {palette: ['1f8d49']}, 'MapBiomas forest 2020');
Map.addLayer(plant2020.selfMask(), {palette: ['d6a000']}, 'MapBiomas plantation 2020');
Map.addLayer(stablePlant.selfMask(), {palette: ['ff0000']}, 'Stable plantation 2016–23', false);
Map.addLayer(south.style({color: 'black', fillColor: '00000000'}), {}, 'States');

// ---------- 7. How much forest vs. plantation per state? (coarse, fast) ----------
var areaKm2 = ee.Image.pixelArea().divide(1e6);
var areas = ee.Image.cat(areaKm2.updateMask(forest2020).rename('forest_km2'),
                         areaKm2.updateMask(plant2020).rename('plantation_km2'))
  .reduceRegions({collection: south, reducer: ee.Reducer.sum(), scale: 300, tileScale: 4});
print('Forest vs plantation area 2020 (km², approx):',
  areas.select(['ADM1_NAME', 'forest_km2', 'plantation_km2']));

// ---------- 8. Click tool: inspect any pixel ----------
Map.onClick(function (c) {
  var pt = ee.Geometry.Point([c.lon, c.lat]);
  print('––––– Point ' + c.lon.toFixed(4) + ', ' + c.lat.toFixed(4) + ' –––––');
  print('MapBiomas class 2020 (3=forest, 9=plantation):',
    lc2020.reduceRegion(ee.Reducer.first(), pt, 30).get('classification_2020'));
  print('MapBiomas class history 2016–2023:', stack.reduceRegion(ee.Reducer.first(), pt, 30));
  print('Canopy height (m):', height.reduceRegion(ee.Reducer.first(), pt, 10));
  print(ui.Chart.image.series(ndvi.filterBounds(pt), pt, ee.Reducer.mean(), 10)
    .setOptions({title: 'Sentinel-2 NDVI 2019–2024', pointSize: 2, lineWidth: 0}));
});

// ---------- 9. (When Cropin's false-positive points arrive) ----------
// Upload their CSV/shapefile as an asset (Assets → New → CSV/Shapefile), then:
// var fp = ee.FeatureCollection('projects/YOUR_PROJECT/assets/cropin_false_positives');
// Map.addLayer(fp, {color: 'magenta'}, 'Cropin false positives');
// var fpClasses = mb.select('classification_2020').sampleRegions({collection: fp, scale: 30});
// print('MapBiomas 2020 class at false-positive points:',
//       fpClasses.aggregate_histogram('classification_2020'));
// → share of points MapBiomas already calls plantation (9) vs forest (3) = your first benchmark

// ---------- 10. Deepak's data (upload the two zips: Assets → New → Shape files) ----------
// var qc  = ee.FeatureCollection('projects/YOUR_PROJECT/assets/cropin_qc');           // 433 polygons, qc_label
// var ref = ee.FeatureCollection('projects/YOUR_PROJECT/assets/mapbiomas_ref_south'); // 2,290 labelled points
// Map.addLayer(ref.filter(ee.Filter.eq('label','forest')),     {color:'1f8d49'}, 'Ref points: forest');
// Map.addLayer(ref.filter(ee.Filter.eq('label','plantation')), {color:'d68a00'}, 'Ref points: plantation');
// Map.addLayer(qc.filter(ee.Filter.eq('qc_label','plantation')), {color:'ff00ff'}, 'Cropin QC: plantation');
// Map.addLayer(qc.filter(ee.Filter.eq('qc_label','forest')),     {color:'00ffff'}, 'Cropin QC: forest');
//
// First benchmark – what does the MapBiomas MAP say inside Cropin's QC polygons?
// var qcMB = mb.select('classification_2020').reduceRegions({
//   collection: qc, reducer: ee.Reducer.mode(), scale: 30});
// print('QC plantation polygons, MapBiomas 2020 class:',
//   qcMB.filter(ee.Filter.eq('qc_label','plantation')).aggregate_histogram('mode'));
// print('QC forest polygons, MapBiomas 2020 class:',
//   qcMB.filter(ee.Filter.eq('qc_label','forest')).aggregate_histogram('mode'));
