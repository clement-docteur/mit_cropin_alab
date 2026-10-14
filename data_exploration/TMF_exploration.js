// Specify regions to include the 3 states
var region = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Brazil'))
  .filter(ee.Filter.inList('ADM1_NAME', ['Parana', 'Santa Catarina', 'Rio Grande Do Sul']));
  
// ------- Visualisation of the Tropical Moist Forest (TMF) dataset -----
// ------- Joint Research Center (JRC) of the European Commission

// Convert RGB to Hex codes

function rgb(r,g,b){var bin = r << 16 | g << 8 | b; return (function(h){  return new Array(7-h.length).join("0")+h;
          })(bin.toString(16).toUpperCase());}


// ------------------------- Transition layer 

// Symbology and legend for the Transition map - Sub types
var PALETTETransitionSubtypes = [
rgb(0,80,0), rgb(10,100,10), rgb(10,90,60), rgb(0,0,0),rgb(0,0,0), rgb(0,0,0), rgb(0,0,0),  rgb(0,0,0), rgb(0,0,0),  rgb(0,0,0), rgb(0,0,0),
rgb(30,120,0), rgb(80,150,0), rgb(100,160,40), rgb(120,170,40), rgb(100,160,40), rgb(120,170,40), rgb(0,0,0), rgb(0,0,0),rgb(0,0,0),rgb(0,0,0),
rgb(185,200,60),  rgb(200,230,60), rgb(210,250,60), rgb(0,0,0), rgb(0,0,0),  rgb(0,0,0), rgb(0,0,0), rgb(0,0,0), rgb(0,0,0), rgb(0,0,0),
rgb(255,240,160), rgb(255,150,8), rgb(0,0,0), rgb(0,0,0),rgb(0,0,0), rgb(0,0,0), rgb(0,0,0), rgb(0,0,0), rgb(0,0,0), rgb(0,0,0),                       
rgb(250,60,10), rgb(170,80,10), rgb(140,100,30),  rgb(140,120,60), rgb(0,0,0), rgb(0,0,0),rgb(0,0,0),  rgb(0,0,0),   rgb(0,0,0), rgb(0,0,0),
rgb(40,100,50), rgb(80,150,0),rgb(200,230,60),rgb(210,250,60),rgb(255,230,110), rgb(255,60,10), rgb(155,105,70),rgb(0,0,0), rgb(0,0,0), rgb(0,0,0),
rgb(0,50,150),rgb(0,150,200), rgb(0,160,150), rgb(0,210,210), rgb(0,0,0), rgb(0,0,0), rgb(0,0,0), rgb(0,0,0),  rgb(0,0,0), rgb(0,0,0),  
rgb(51,99,51),rgb(98,161,80), rgb(188,209,105), rgb(255,228,148), rgb(250,180,150), rgb(204,163,163), rgb(0,0,0), rgb(0,0,0), rgb(0,0,0), rgb(0,0,0), 
rgb(255,255,255), rgb(237,255,215), rgb(224,250,157), rgb(214,250,188)
];

// Visualization of the Transition map - Sub types
var TransitionMap=ee.ImageCollection('projects/JRC/TMF/v1_2025/TransitionMap_Subtypes')
                  .filterBounds(region)
                  .mosaic(); 

Map.centerObject(region);   
Map.addLayer(TransitionMap.updateMask(TransitionMap).clip(region),{'min':10, 'max': 94, 'palette': PALETTETransitionSubtypes}, "JRC - Transition Map – Sub types - v1 2025",true);

// Legend Transition Map - Sub types (v1-2025)

// value 10. Undisturbed Tropical Moist Forest (TMF) 
// value 11. Bamboo-dominated forest
// value 12. Undisturbed mangroves

// value 21. Degraded forest with short-duration disturbance (started before 2015)
// value 22. Degraded forest with short-duration disturbance (started in 2016-2024)
// value 23. Degraded forest with long-duration disturbance (started before 2015)
// value 24. Degraded forest with long-duration disturbance (started in 2016-2024)
// value 25. Degraded forest with 2/3 degradation periods (last degradation started before 2015)
// value 26. Degraded forest with 2/3 degradation periods (last degradation started in 2016-2024)

// value 31. Old forest regrowth (disturbed before 2006)
// value 32. Young forest regrowth (disturbed in 2006-2015)
// value 33. Very young forest regrowth (disturbed in 2016-2022)

// value 41. Deforestation started before 2015
// value 42. Deforestation started in 2016-2022

// value 51. Deforestation* started in 2023 
// value 52. Deforestation* started in 2024 
// value 53. Deforestation* started in 2025  
// value 54. Degradation* started in 2025 

// value 61. Degraded Mangroves (started before 2016)
// value 62. Mangrove recently degraded (2016 -2024)
// value 63. Mangrove regrowing (at least 10 years - 2016-2025)
// value 64. Mangrove regrowing (at least 3 years - 2023-2025)
// value 65. Mangrove deforested (started before 2015)
// value 66. Mangrove deforested (started in 2016-2022)
// value 67. Mangrove recently disturbed (started in 2023-2025)

// value 71. Permanent water (Pekel et al. 2016 & updates for years 2015-2021)     
// value 72. Seasonal water  (Pekel et al. 2016 & updates for years 2015-2021)     
// value 73. Deforestation to permanent water 
// value 74. Deforestation to seasonal water 

// value 81. Old Plantation
// value 82. Plantation regrowing (disturbed before 2016) 
// value 83. Plantation regrowing (disturbed in 2016-2022)  
// value 84. Conversion to plantation (deforestation started before 2015)
// value 85. Conversion to plantation (deforestation started in 2016-2022)
// value 86. Recent Conversion to plantation (started in 2023-2025)

// value 91.  Other land cover without afforestation  
// value 92.  Young afforestation (between 3 and 9 years of regrowth)
// value 93.  Old afforestation (between 10 and 20 years of regrowth)
// value 94.  Water converted recently into forest regrowth (at least 3 years)

