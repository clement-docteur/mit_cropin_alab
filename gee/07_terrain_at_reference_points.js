// #####################################################################
// 07_terrain_at_reference_points.js  (Meera, Nov 2026)
// WHAT IT DOES
//   In the blind map check (gee/06), winter SAG in the SC test area looked like a hillshade:
//   the sun is low in winter, so slopes facing away from it are in shadow. Points where winter
//   added plantation in SC were mostly native forest. This script reads terrain at the 2,290
//   MapBiomas reference points so src/slope_winter_check.py can test whether the winter gain
//   shrinks, or turns into false plantation calls, on steep or shaded ground.
// ASSETS NEEDED
//   projects/cropin-510901/assets/mapbiomas_ref_south  (2,290 points: id, label, state, high_conf)
//   Public: NASA SRTM 30 m (USGS/SRTMGL1_003).
// HOW TO RUN
//   Paste into a new script in the Code Editor (project cropin-510901) and press Run. In the
//   Tasks tab, click RUN on the one task. The file lands in Google Drive/cropin_exports. Copy it to
//   data/gee_exports/ref_terrain.csv.
// OUTPUT (one row per reference point)
//   id, label, state, elev_m, slope_deg, aspect_deg (0 = north, 90 = east),
//   hs_winter, hs_summer  hillshade 0-255 (0 = full shadow, 255 = facing the sun) for the sun
//                         position at the Sentinel-2 overpass (about 10:30 local) in June and in
//                         December at about 27 S. Winter: sun 33 deg high, from the north-east (35 deg).
//                         Summer: sun 68 deg high, from the east (85 deg). Fixed values, so these are
//                         approximate, but enough to rank points from shaded to sunlit.
//   relief_m              max minus min elevation within 100 m (how rugged the surroundings are)
// #####################################################################

var ref = ee.FeatureCollection('projects/cropin-510901/assets/mapbiomas_ref_south');
var dem = ee.Image('USGS/SRTMGL1_003').select('elevation');
var terrain = ee.Terrain.products(dem);           // adds slope, aspect (degrees)

var img = dem.rename('elev_m')
  .addBands(terrain.select('slope').rename('slope_deg'))
  .addBands(terrain.select('aspect').rename('aspect_deg'))
  .addBands(ee.Terrain.hillshade(dem, 35, 33).rename('hs_winter'))
  .addBands(ee.Terrain.hillshade(dem, 85, 68).rename('hs_summer'))
  .addBands(dem.reduceNeighborhood(ee.Reducer.minMax(), ee.Kernel.circle(100, 'meters'))
            .expression('b(1) - b(0)').rename('relief_m'));

var out = img.sampleRegions({collection: ref, properties: ['id', 'label', 'state'], scale: 30, geometries: false});
print('points sampled (expect 2,290):', out.size());
print('example row:', out.first());

Export.table.toDrive({
  collection: out, description: 'ref_terrain', folder: 'cropin_exports', fileFormat: 'CSV',
  selectors: ['id', 'label', 'state', 'elev_m', 'slope_deg', 'aspect_deg', 'hs_winter', 'hs_summer', 'relief_m']
});
