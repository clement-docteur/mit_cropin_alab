var year = 2025

var Palette = require('users/mapbiomas-global/LULC:LULC_palette.js');
var vis_LULC = Palette.get('vis_LULC');

// MapBiomas land use and land cover
var collection = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1')
Map.addLayer(collection.select('classification_'+year), vis_LULC, 'MapBiomas 10 Meters Col. 4 - '+ year, true);

Map.setCenter(-51.0, -15.0, 4)