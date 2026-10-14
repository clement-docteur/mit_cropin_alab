var points  = ee.FeatureCollection('projects/cropin-510901/assets/mapbiomas_85k_points_validation');
Map.addLayer(points.style({color: 'white', pointSize: 3, width: 1}),
             {}, 'Validation points', true);  

print(points.first());