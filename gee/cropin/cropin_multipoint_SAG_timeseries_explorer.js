// =====================================================================
// FROM CROPIN (Uddesh / Deepak), shared by email on 2026-10-28 as a GEE snapshot:
//   https://code.earthengine.google.com/d243b59bbc051a59d96980776bc0bd9a
// Click-to-compare SAG time series (10-day composites, May 2022–May 2025).
// Copied verbatim below this header; logic NOT modified. See CONTEXT.md §2e.
// =====================================================================

//// ****************************      Region Check             **********************

//// Add google map or Qgis lat long point 
var google_map_point = [-25.0770160,-50.5611659]
var point = ee.Geometry.Point([google_map_point[1], google_map_point[0]]) 
Map.addLayer(point, {color:"yellow"},"Selected Point" )
Map.centerObject(point,17)

//// ****************************      Define Functions           **********************
var startDate = '2022-05-01'; // Defined start date
var endDate = '2025-05-01';   // Defined end date

var addNDVI = function(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}; 

// Use Google's Cloud Score+ Mask
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
var csPlusBands = csPlus.first().bandNames();

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


// Optimized Image Collection Generator (No AOI needed, GEE filters by chart regions automatically)
function get_S2_composite(startdate, enddate, interval) {
  var total_days = ee.Date(enddate).difference(ee.Date(startdate), "day");
  var s2_dataset = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate(startdate, enddate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
    .linkCollection(csPlus, csPlusBands)
    .map(maskLowQA)
    .map(addGreenAngle); // Fixed: Replaced undefined 'addGreenAngle'
        
  // Creating mosaic for every 10-day interval
  var days = ee.List.sequence(0, total_days, interval);
  var composites = days.map(function(d) {
    var t1 = ee.Date(startdate).advance(d, 'days');
    var t2 = t1.advance(interval, 'days');
  
    return s2_dataset.filterDate(t1, t2)
          .median()
          .set('system:time_start', t1.millis()) // Required for charting
          .set('system:id', t1.format('YY-MM-dd'));
  });

  return ee.ImageCollection(composites)
    // Filter out empty composite periods to prevent chart rendering errors
    .map(function(img) { return img.set('Bandcount', img.bandNames().length()); })
    .filterMetadata("Bandcount", "greater_than", 0)
    .select('SAG'); // Select only the NDVI band
}

// Generate the collection ONCE globally, not on every click
var image_coll = get_S2_composite(startDate, endDate, 10);
 
//// ****************************     Point NDVI Check      **********************
var panel = ui.Panel();
panel.style().set('width', '600px');

var intro = ui.Panel([
  ui.Label({
    value: 'SAG Chart Inspector',
    style: { fontSize: '20px', fontWeight: 'bold' }
  }),
  ui.Label('Click multiple points on the map to compare them.')
]);
panel.add(intro);

var lon = ui.Label();
var lat = ui.Label();
panel.add(ui.Panel([lon, lat], ui.Panel.Layout.flow('horizontal')));

// Store all clicked points
var clickedPoints = [];

// Function to update the chart with all clicked points
function updateChart() {
  // Convert the client-side point list to an EE FeatureCollection
  var pointFeatures = clickedPoints.map(function(p, index) {
    return ee.Feature(
      ee.Geometry.Point([p.lon, p.lat]),
      { 'id': 'Point ' + (index + 1) }
    );
  });
  
  var pointsFC = ee.FeatureCollection(pointFeatures);

  // OPTIMIZATION: Use built-in seriesByRegion instead of mapping/reducing manually
  var ndviChart = ui.Chart.image.seriesByRegion({
    imageCollection: image_coll,
    regions: pointsFC,
    reducer: ee.Reducer.mean(),
    band: 'SAG',
    scale: 10,
    xProperty: 'system:time_start',
    seriesProperty: 'id'
  });

  ndviChart.setOptions({
    title: 'SAG Over Time - All Selected Points',
    hAxis: { title: 'Date', gridlines: { count: 5 } },
    vAxis: { title: 'SAG' },
    lineWidth: 1,
    pointSize: 3,
    interpolateNulls: true
  });

  // Put the updated chart in the panel (replaces the previous chart at index 2)
  panel.widgets().set(2, ndviChart);
}

// Map click event
Map.onClick(function(coords) {
  lon.setValue('lon: ' + coords.lon.toFixed(5));
  lat.setValue('lat: ' + coords.lat.toFixed(5));

  // Store the clicked coordinate
  clickedPoints.push({
    lon: coords.lon,
    lat: coords.lat
  });

  // Add a red point to the map
  var point = ee.Geometry.Point([coords.lon, coords.lat]);
  var dot = ui.Map.Layer(point, {color: 'FF0000'}, 'Point ' + clickedPoints.length);
  
  Map.layers().add(dot);
  updateChart();
});

Map.style().set('cursor', 'crosshair');
ui.root.insert(0, panel);
Map.setOptions('SATELLITE');
