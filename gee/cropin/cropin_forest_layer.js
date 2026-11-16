// =====================================================================
// FROM CROPIN (Uddesh / Deepak), shared by email on 2026-10-28 as a GEE snapshot:
//   https://code.earthengine.google.com/6a85964d7c9cc0982dfd60c98e41cb9a
// Cropin's forest-layer recipe (ensemble of Hansen, JRC, MapBiomas, TMF + MapBiomas 10m + Meta height).
// Copied verbatim below this header; logic NOT modified. See CONTEXT.md §2e.
// =====================================================================

// ======================================================
// # 0. Defining AOI and Base year of Forest Layer
// ======================================================

// --- Manual Point Selection for centering/testing ---
var google_map_point = [-30.7519017,-52.2737933]; // [Latitude, Longitude]
// Create a Geometry object: EE requires [Longitude, Latitude]
var selected_point_geom = ee.Geometry.Point([google_map_point[1], google_map_point[0]]); 

Map.addLayer(selected_point_geom, {color:"yellow"},"Selected Point") ;
Map.centerObject(selected_point_geom,15)
Map.setOptions("SATELLITE");


var TARGET_YEAR = 2020;

// ======================================================
// # 1. Hansen Global Forest Change Dataset 
// This section filters forest areas using the historical tree cover data.
// ======================================================
var gfc_hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12");

// Select the 2000 tree cover layer
var tree_cover_hansen = gfc_hansen.select('treecover2000');

// Identify years where loss occurred (lossyear > 0 and <= 20).
// Note: The unmask(-99) handles null values in the 'lossyear' band.
var forest_loss = gfc_hansen.select('lossyear')
    .unmask(-99) // Replace masked values with -99 for comparison safety
    .gt(0)        // Must have experienced some loss year (> 0)
    .and(gfc_hansen.select('lossyear').unmask(-99).lte(20)); // And within the last 20 years

// Define 'Hansen Forest': areas with high tree cover (>= 30%) AND no recorded recent loss event.
var hansen_forest = tree_cover_hansen.gte(30)
    .and(forest_loss.not()) // Use .not() to mask out areas that experienced forest loss
    .rename('hansen_forest');

// ======================================================
// # 2. MapBiomas LULC Datasets (Land Cover Classification)
// Extracts Forest and Planatation data different resolutions/sources.
// ======================================================
var mapbiomas_lulc_10m = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1');
var mapbiomas_lulc_10m_beta = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection_S2_beta/collection_LULC_S2_beta');
var mapbiomas_lulc_30m = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_coverage_v2');

// Select classification layers for the target year (using 10m resolution datasets)
var class_year_10 = mapbiomas_lulc_10m.select('classification_' + TARGET_YEAR);
var class_year_10m_beta = mapbiomas_lulc_10m_beta.select('classification_' + TARGET_YEAR);
var class_year_30 = mapbiomas_lulc_30m.select('classification_' + TARGET_YEAR);

// Define forest indicators (code 3 for Forest)
var mbi_forest_30m = class_year_30.eq(3).rename('mbi_forest'); // Primary source for forest cover (30m res)

// Define plantation indicators (codes 9 for plantations)
var mbi_plantation10 = class_year_10.eq(9);        // 10m resolution plantation
var mbi_forest10_beta = class_year_10m_beta.eq(3).rename('mbi_forest_10'); // 10m beta forest
var mbi_plantation10_beta = class_year_10m_beta.eq(9);   // 10m beta plantation

// Combined mask for plantations (to exclude these areas later)
var combined_plantations = mbi_plantation10_beta.or(mbi_plantation10);


// ======================================================
// # 3. JRC Global Forest Cover Dataset (GFC2020)
// Uses the Joint Research Centre dataset for a broad estimate of forest cover.
// ======================================================
var jrc_gfc_2020 = ee.ImageCollection('JRC/GFC2020/V2').mosaic();

// Filter: Keep pixels with GFC value > 0.5 and mask out others
var jrc_forest = jrc_gfc_2020.gt(0.5).selfMask().rename('jrc_forest');


// ======================================================
// # 4. Tropical Moist Forest (TMF) Dataset
// Uses a specialized dataset for tropical forest assessment.
// ======================================================
var tmf_annual_changes = ee.ImageCollection('projects/JRC/TMF/v1_2024/AnnualChanges').mosaic();

// Select the TMF layer for the target year (assuming 'Dec' prefix)
var tmf_year_layer = tmf_annual_changes.select('Dec' + TARGET_YEAR);

// Define 'TMF Forest': Areas where change is minor (<= 2) or stable/high (== 4).
var tmf_forest = tmf_year_layer.lte(2).or(tmf_year_layer.eq(4)).selfMask().rename('tmf_forest');

// ======================================================
// # 5. Weighted Ensemble Calculation (Averaging the Sources)
// Determines weights dynamically based on the target year.
// ======================================================

var ee_number = ee.Number(TARGET_YEAR);

// Dynamic Weight Assignment: Checks if the year is 2020 to set different weights for each dataset.
// Structure: [Weight_Hansen, Weight_JRC, Weight_MapBiomas_30m, Weight_TMF]
var dynamic_weights = ee.Algorithms.If(ee_number.eq(2020), 
                                        [2, 2, 3, 3], // Weights for 2020 (Example)
                                        [2, 0, 3, 3]);// Fallback weights

var weight_list = ee.List(dynamic_weights);

// Extract individual weights
var w_hansen = weight_list.get(0);
var w_jrc = weight_list.get(1);
var w_mbi = weight_list.get(2);
var w_tmf = weight_list.get(3);

// Calculate the sum of all weights (used for normalization)
var total_weight_sum = weight_list.reduce(ee.Reducer.sum());


// Apply weights to each binary forest mask (1=Forest, 0=Non-forest/Masked).
var weighted_hansen = hansen_forest.unmask(0).multiply(ee.Number(w_hansen));
var weighted_jrc = jrc_forest.unmask(0).multiply(ee.Number(w_jrc));
var weighted_mbi = mbi_forest_30m.unmask(0).multiply(ee.Number(w_mbi));
var weighted_tmf = tmf_forest.unmask(0).multiply(ee.Number(w_tmf));

// Combine the four weighted images using addition and normalize by dividing by the total weight sum.
var image_sum_weighted = weighted_hansen.add(weighted_jrc)
    .add(weighted_mbi).add(weighted_tmf);

var final_ensemble_image = image_sum_weighted.divide(ee.Number(total_weight_sum))
    .rename('forest_score');

// Initial mask: Exclude areas identified as plantations (since we are focusing on natural forest/secondary growth).
var masked_by_plantations = final_ensemble_image.updateMask(combined_plantations.unmask().eq(0));


// ======================================================
// # 6. Canopy Height Constraint (Refining the Mask)
// Filters out potential forest pixels that lack structural height evidence.
// ======================================================

var canopy_height_collection = ee.ImageCollection("projects/sat-io/open-datasets/facebook/meta-canopy-height").mosaic();

// Filter: Keep pixels with detected canopy height (>= 1 meter)
var height_constraint = canopy_height_collection.gte(4).rename('height_mask');

// Apply all remaining masks:
// 1. Mask by the best MapBiomas forest source (10m beta mask, for high resolution)
// 2. Mask by the Canopy Height constraint
var final_forest_layer = masked_by_plantations
    .updateMask(mbi_forest10_beta) // Use highest resolution source for masking refinement
    .updateMask(height_constraint);


// ======================================================
// # 7. Visualization and Map Display
// ======================================================

// Add the final composite forest layer to the map
Map.addLayer(final_forest_layer.gte(0.75).selfMask(), {min:0, max:1, palette:["darkgreen"]}, 'Final Forest Mask', 1);
Map.addLayer(final_forest_layer, {min:0, max:1, palette:["blue", "green","red"]}, 'Final Forest Ensemble', 0);

