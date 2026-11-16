// =====================================================================
// FROM CROPIN (Uddesh / Deepak), shared by email on 2026-10-28 as a GEE snapshot:
//   https://code.earthengine.google.com/96c128110c3ee110203d96734a1083c6
// Multi-year SAG plantation probability (Bayesian update over Oct–Jan, 2019–2025) for a 6 km box.
// Copied verbatim below this header; logic NOT modified. See CONTEXT.md §2e.
// =====================================================================

//// ****************************      Selecting  AOI            **********************

/// Draw rectangle in selected cropland area

// var aoi = region 
//// Add google map or Qgis lat long point 
// var google_map_point = [-30.41657, -51.75877]
// var google_map_point = [-28.249922,-49.358079]
// var google_map_point = [-27.6237375,-50.0951435]
var google_map_point = [-25.0770160,-50.5611659]
 

var point = ee.Geometry.Point([google_map_point[1], google_map_point[0]]) 

Map.addLayer(point, {color:"yellow"},"Selected Point" )
Map.centerObject(point,15)
Map.setOptions("SATELLITE")

var aoi = point.buffer(3000).bounds()
Map.addLayer(aoi,{color:"green"},"Cropland aoi",0)


// ***************   Defining initial variables bands, dates and aoi
var bands = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"];
var startdate = "2020-10-01"
var enddate = "2021-01-31"
var total_days = ee.Date(enddate).difference(ee.Date(startdate),"day")
var interval = 10;
var rgbVis = {min: 0.0, max: 3000, bands: ['B4', 'B3', 'B2']};
var palette1 = ["#a6cee3","#1f78b4"]
var palette2 = ["#a6cee3","#6a3d9a"]
var palette3 = ["#a6cee3","#e31a1c"]

//// ****************************      Define Functions           **********************

// Function for adding NDVI
var addNDVI = function(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}; 

//Function to add green NDVIa
var addGNDVI = function(image) {
  var gndvi = image.normalizedDifference(['B3', 'B4']).rename('GNDVI');
  return image.addBands(gndvi);

}; 

// Function to mask pixels with low CS+ QA scores.
function maskLowQA(image) {
  var qaBand = 'cs';
  var clearThreshold = 0.6;
  var mask = image.select(qaBand).gte(clearThreshold);
  return image.updateMask(mask);
}

function addGreenAngle(s2_dataset){
  
    // Extract the Y values (Reflectance)
  var yB = s2_dataset.select('B2').divide(10000); // Blue
  var yG = s2_dataset.select('B3').divide(10000); // Green
  var yR = s2_dataset.select('B4').divide(10000); // Red
  
  var yRE = s2_dataset.select('B8A').divide(10000); // Red Edge
  
  // 3. Define the X values (Center wavelengths in micrometers)
  var xB = 0.4924; // Blue
  var xG = 0.5598; // Green
  var xR = 0.6646; // Red
  
  // 4. Calculate Vectors from Green (Vertex) to Blue and Red
  // Vector 1: Green -> Blue
  var dx1 = xB - xG;                   // JavaScript constant
  var dy1 = yB.subtract(yG);           // GEE Image
  
  // Vector 2: Green -> Red
  var dx2 = xR - xG;                   // JavaScript constant
  var dy2 = yR.subtract(yG);           // GEE Image
  
  // 5. Compute the Dot Product ( dx1*dx2 + dy1*dy2 )
  var dotProduct = dy1.multiply(dy2).add(dx1 * dx2);
  
  // 6. Compute Vector Magnitudes
  var mag1 = dy1.pow(2).add(Math.pow(dx1, 2)).sqrt();
  var mag2 = dy2.pow(2).add(Math.pow(dx2, 2)).sqrt();
  
  // 7. Calculate the Angle using the arccosine of (Dot / (Mag1 * Mag2))
  var cosTheta = dotProduct.divide(mag1.multiply(mag2));
  
  // Clamp between -1 and 1 to prevent NaN errors from floating point inaccuracies
  cosTheta = cosTheta.clamp(-1.0, 1.0);
  
  // Get the angle in radians, then convert to degrees
  var angleRad = cosTheta.acos();
  var angleDeg = angleRad.multiply(180 / Math.PI).rename('SAG');
    
  var band_value_mask = yG.lte(0.05).and(yR.lte(0.05)).and(yB.lte(0.05))
  
  return s2_dataset.addBands(angleDeg)
                    .mask(band_value_mask)
  
}


// Gaussian PDF: N(x; mean, std)
function gaussianPDF(img, mean, std) {
  var exponent = img.subtract(mean).pow(2)
                      .multiply(-1)
                      .divide(2 * std * std);
  var coeff = 1 / (std * Math.sqrt(2 * Math.PI));
  return exponent.exp().multiply(coeff);
}


// 3. This is the function to pass to iterate().
// As posterior images are computed, add them to the list.
var accumulate = function(image, list) {
  // Get the latest posterior image from the end of the list with get(-1).
  var previous = ee.Image(ee.List(list).get(-1));
  var img = ee.Image(image);

  var likPlantation    = gaussianPDF(img, PlantationMean, PlantationStd);
  var likNonPlantation = gaussianPDF(img, nonPlantationMean, nonPlantationStd);

  var priorPlantation    = previous;
  var priorNonPlantation = ee.Image(1).subtract(previous);

  var numerator   = likPlantation.multiply(priorPlantation);
  var denominator = numerator.add(likNonPlantation.multiply(priorNonPlantation));
  
  var newPosterior = numerator.divide(denominator);

  // Where this date's pixel is masked (cloud/no data), carry forward the
  // previous posterior instead of updating.
  // var validMask = img.mask();
  newPosterior = newPosterior.unmask(previous)
                  // .where(validMask.not(), previous);
  
  newPosterior = newPosterior.clamp(0.1, 0.9).rename('posterior').float()
    // Propagate the timestamp to the new image so it charts correctly.
    .set('system:time_start', img.get('system:time_start'));

  // Return the list with the new posterior inserted.
  return ee.List(list).add(newPosterior);
};


// Use Google's Cloud Score+ Mask
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var csPlusBands = csPlus.first().bandNames();

// ====================================================
// 6. Bayesian sequential update over the SAG time series
// ====================================================

// Class-conditional Gaussian params
var PlantationMean = 160,  PlantationStd = 9;
var nonPlantationMean = 150, nonPlantationStd = 9;


var images = [];

for (var i = 2019; i < 2026; i++) {
  
  var startdate = i + '-10-01';
  var enddate = (i + 1) + '-01-31';
  
  var s2_dataset_main = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filter(ee.Filter.date(startdate, enddate))
              .filterBounds(aoi)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
              .linkCollection(csPlus, csPlusBands)
              .map(maskLowQA)
              
  var s2_dataset = s2_dataset_main.map(addGreenAngle)  
                             .select('SAG')
                             .sort('system:time_start');

  // Initial prior: 0.5 Plantation / 0.5 non-Plantation everywhere in the AOI
  var initPrior = ee.Image.constant(0.5).rename('posterior').clip(aoi);
  
  var s2_datasetList = s2_dataset.toList(s2_dataset.size());
  
  // Sequentially fold Bayes' rule over each observation:
  // posterior_t becomes the prior for observation_t+1
  var posteriorFinal = ee.Image(s2_datasetList.iterate(function(imgObj, prevPosterior) {
    var img = ee.Image(imgObj);
    var prevPost = ee.Image(prevPosterior);
  
    var likPlantation    = gaussianPDF(img, PlantationMean, PlantationStd);
    var likNonPlantation = gaussianPDF(img, nonPlantationMean, nonPlantationStd);
  
    var priorPlantation    = prevPost;
    var priorNonPlantation = ee.Image(1).subtract(prevPost);
  
    var numerator   = likPlantation.multiply(priorPlantation);
    var denominator = numerator.add(likNonPlantation.multiply(priorNonPlantation));
   
    var newPosterior = numerator.divide(denominator);

  // Where this date's pixel is masked (cloud/no data), carry forward the
  // previous posterior instead of updating.
  // var validMask = img.mask();
    var newPosterior = newPosterior.unmask(prevPost)
                            // .where(validMask.not(), prevPost);

  return newPosterior.clamp(0.1, 0.9).rename('posterior');
}, initPrior));

  var posteriorFinal = posteriorFinal
    .set('year', i)
    .set('start_date', startdate)
    .set('end_date', enddate)
    .set('system:index',ee.Number(i).format() );

  images.push(posteriorFinal);
}

var imageCollection = ee.ImageCollection.fromImages(images);

// Add each year's image to the map
for (var i = 2019; i < 2026; i++) {
  
  var image = imageCollection
    .filter(ee.Filter.eq('year', i))
    .first();

  Map.addLayer(
    image,
    {min: 0, max: 1, palette: ['blue', 'green', 'yellow', 'red']},                 // visualization parameters
    'Year ' + i,
    false              // initially turned off
  );
}


var mean_posterior = imageCollection.mean()
var canopyHeight = ee.ImageCollection("projects/sat-io/open-datasets/facebook/meta-canopy-height").mosaic();
var treenotree = canopyHeight.gte(4)

Map.addLayer( mean_posterior.updateMask(treenotree),  {min: 0, max: 1, palette: ['blue', 'green', 'yellow', 'red']},"7 year mean posterior",  false);
Map.addLayer( mean_posterior.gte(0.8).updateMask(treenotree).selfMask(),  {min: 0, max: 1, palette: ['white', 'red']},"Plantation mask",  false);


// # Map-Biomass Plantation layer

var year = 2020
var mBiomass_lulc_10 = ee.Image('projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1')
var mBiomass_lulc_10_beta = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection_S2_beta/collection_LULC_S2_beta')

var class_year_10 = mBiomass_lulc_10.select('classification_'+year)
var class_year_10_beta = mBiomass_lulc_10_beta.select('classification_'+year)

var mBiomass_plantation10 = class_year_10.eq(9)
var mBiomass_plantation10_beta = class_year_10_beta.eq(9)

var mBiomass_plantation = mBiomass_plantation10_beta.or(mBiomass_plantation10)

Map.addLayer(mBiomass_plantation.updateMask(treenotree).selfMask(), {min: 0, max: 1, palette: ['pink']}, 'MB Plantation',0);


// Exporting posterior image

// var posterior_images = imageCollection.toBands()
// Export.image.toDrive({image:posterior_images, 
//                       description: "Plantation_Multiyear_posterior", 
//                       folder:"GEE",
//                       // fileNamePrefix, dimensions, 
//                       region:aoi, 
//                       scale:10, 
//                       crs:"EPSG:4326",
//                       // crsTransform, maxPixels, shardSize, fileDimensions, skipEmptyTiles, fileFormat, formatOptions,
//                       priority:9900
  
// })  