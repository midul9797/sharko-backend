import joblib
import numpy as np
from datetime import datetime
import pandas as pd
import json
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import unary_union
import random
from math import sqrt
import warnings
import geojson
from sklearn.cluster import DBSCAN
warnings.filterwarnings('ignore', category=UserWarning)

def load_models():
    """Load all required models"""
    models = {
        'sst': joblib.load('models/sst.joblib'),
        'ssh': joblib.load('models/ssh.joblib'),
        'ssha_scaler': joblib.load('models/ssha_model_robust_dependent_tuned.joblib')['scaler'],
        'chla': joblib.load('models/chloro_model_log_independent_tuned.joblib'),
        'shark': joblib.load('models/shark_presence.joblib')
    }
    return models

def prepare_temporal_features(dates):
    """Convert dates to cyclical features"""
    day_of_years = []
    day_sins = []
    day_coss = []

    for date_str in dates:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        day_of_year = date_obj.timetuple().tm_yday
        day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
        day_cos = np.cos(2 * np.pi * day_of_year / 365.25)

        day_of_years.append(day_of_year)
        day_sins.append(day_sin)
        day_coss.append(day_cos)

    return day_of_years, day_sins, day_coss

def calculate_polygon_area_km2(polygon):
    """Calculate polygon area in km² with fallback projections"""
    gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs='EPSG:4326')

    # Try different projections in order of preference
    projections_to_try = [
        'EPSG:3857',  # Web Mercator (widely available)
        '+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',  # Mollweide
        '+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',  # Eckert IV
    ]

    for proj in projections_to_try:
        try:
            gdf_projected = gdf.to_crs(proj)
            area_m2 = gdf_projected.geometry.area.iloc[0]
            area_km2 = area_m2 / 1_000_000  # Convert to km²
            return area_km2
        except Exception as e:
            continue

    # Fallback: approximate area using spherical geometry
    # Simple approximation for small areas
    bounds = polygon.bounds
    lat_center = (bounds[1] + bounds[3]) / 2

    # Convert degrees to approximate km (rough approximation)
    lat_km = (bounds[3] - bounds[1]) * 111.32  # 1 degree lat ≈ 111.32 km
    lon_km = (bounds[2] - bounds[0]) * 111.32 * np.cos(np.radians(lat_center))

    # Very rough area approximation
    approx_area = lat_km * lon_km
    return max(1.0, approx_area)  # Minimum 1 km²

def generate_random_points_in_polygon(polygon, n_points):
    """Generate n random points within a polygon using rejection sampling"""
    points = []
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)

    attempts = 0
    max_attempts = n_points * 1000  # Prevent infinite loops

    while len(points) < n_points and attempts < max_attempts:
        # Generate random point within bounding box
        x = random.uniform(bounds[0], bounds[2])
        y = random.uniform(bounds[1], bounds[3])
        point = Point(x, y)

        # Check if point is within polygon
        if polygon.contains(point):
            points.append((y, x))  # (lat, lon)

        attempts += 1

    return points

def generate_grid_points_in_polygon(polygon, n_points):
    """Generate approximately n points in a grid pattern within polygon"""
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)

    # Calculate grid dimensions
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    # Estimate grid size (assuming square grid)
    grid_size = int(sqrt(n_points))

    # Calculate step sizes
    x_step = width / grid_size
    y_step = height / grid_size

    points = []

    for i in range(grid_size + 1):
        for j in range(grid_size + 1):
            x = bounds[0] + i * x_step
            y = bounds[1] + j * y_step
            point = Point(x, y)

            if polygon.contains(point):
                points.append((y, x))  # (lat, lon)

    # If we have more points than needed, randomly sample
    if len(points) > n_points:
        points = random.sample(points, n_points)
    # If we have fewer points, add random points to reach target
    elif len(points) < n_points:
        additional_needed = n_points - len(points)
        additional_points = generate_random_points_in_polygon(polygon, additional_needed)
        points.extend(additional_points)

    return points

def load_geojson_polygons(geojson_file):
    """Load polygons from GeoJSON file"""
    gdf = gpd.read_file(geojson_file)
    return gdf

def generate_prediction_points(gdf, points_per_polygon=100, method='random', target_date='2023-04-14'):
    """
    Generate prediction points for all polygons in GeoDataFrame

    Parameters:
    gdf: GeoDataFrame containing polygons
    points_per_polygon: Number of points to generate per polygon
    method: 'random', 'grid', or 'adaptive' (adaptive adjusts points based on area)
    target_date: Date for prediction

    Returns:
    DataFrame with columns ['polygon_id', 'lat', 'lon', 'date', 'area_km2']
    """
    all_points = []

    for idx, row in gdf.iterrows():
        polygon = row.geometry

        # Calculate area
        area_km2 = calculate_polygon_area_km2(polygon)

        # Adjust number of points based on method
        if method == 'adaptive':
            # Scale points based on area (1 point per km²)
            n_points = max(10, min(1000, int(area_km2)))  # Between 10 and 1000 points
        else:
            n_points = points_per_polygon

        # Generate points
        if method == 'grid':
            points = generate_grid_points_in_polygon(polygon, n_points)
        else:  # random or adaptive
            points = generate_random_points_in_polygon(polygon, n_points)

        # Add to results
        for lat, lon in points:
            all_points.append({
                'polygon_id': idx,
                'lat': lat,
                'lon': lon,
                'date': target_date,
                'area_km2': area_km2
            })

    return pd.DataFrame(all_points)

def batch_predict(input_data, models):
    """
    Perform batch predictions for shark presence

    Parameters:
    input_data: DataFrame with columns ['lat', 'lon', 'date']
    models: Dictionary containing all loaded models

    Returns:
    DataFrame with all predictions
    """

    # Prepare temporal features
    _, day_sins, day_coss = prepare_temporal_features(input_data['date'])

    # Create base features DataFrame
    base_features = pd.DataFrame({
        'lat': input_data['lat'],
        'lon': input_data['lon'],
        'day_sin': day_sins,
        'day_cos': day_coss
    })

    # Step 1: Predict SST

    
    predictions_sst = models['sst'].predict(base_features)
    ssha_input_features = base_features.copy()
    ssha_input_features['predicted_sst'] = predictions_sst
    # Step 2: Predict SSH (using SST predictions)
    predictions_ssh = models['ssh'].predict(ssha_input_features)
    # Reverse the transformation using the scaler from the bundle to get the real-world value
    

    chloro_input_features = ssha_input_features.copy()
    chloro_input_features['predicted_ssha'] = predictions_ssh
    predicted_chloro_log = models['chla'].predict(chloro_input_features)
    predictions_chla = np.expm1(predicted_chloro_log) # Reverse the log transformation


    # Step 4: Final shark presence prediction
    final_features = pd.DataFrame({
        'lat': input_data['lat'],
        'lon': input_data['lon'],
        'chlor_a': predictions_chla,
        'sst': predictions_sst,
        'ssha': predictions_ssh,
        'day_sin': day_sins,
        'day_cos': day_coss
    })
    predictions_shark = models['shark'].predict(final_features)

    # Combine all results
    results = input_data.copy()
    results['predicted_sst'] = predictions_sst
    results['predicted_ssh'] = predictions_ssh
    results['predicted_chla'] = predictions_chla
    results['predicted_shark_presence'] = predictions_shark
    results['day_sin'] = day_sins
    results['day_cos'] = day_coss

    return results

def predict_shark_presence(geojson_file='sharks_zone2.geojson', points_per_polygon=200, 
                 prediction_date='2030-05-14', point_generation_method='adaptive',
                 epsilon=1, min_samples=5):
    """
    Main function that performs the complete prediction pipeline
    Returns GeoJSON feature collection
    """
    try:
        # Load models
        models = load_models()
        # Load GeoJSON polygons
        gdf = load_geojson_polygons(geojson_file)

        # Generate prediction points
        # prediction_points = generate_prediction_points(
        #     gdf,
        #     points_per_polygon=points_per_polygon,
        #     method=point_generation_method,
        #     target_date=prediction_date
        # )
        # Save generated prediction points to CSV
        # try:
        #     prediction_points.to_parquet('prediction_points.parquet', index=False)
        # except Exception:
        #     pass
        prediction_points = pd.read_parquet('prediction_points.parquet')
        # Run batch predictions
        
        results = batch_predict(prediction_points, models)
        # Convert to DataFrame for clustering
        dframe = pd.DataFrame(results)
        
        
        # Perform DBSCAN Clustering
        db_presence = DBSCAN(eps=epsilon, min_samples=min_samples).fit(dframe[['lon', 'lat']].to_numpy())
        labels = db_presence.labels_

        # Get the unique cluster labels, ignoring noise (-1)
        unique_labels = set(labels)
        unique_labels.discard(-1)

        # Create a polygon for each cluster
        features = []
        for label in unique_labels:
            # Get all points belonging to the current cluster
            cluster_points = dframe[['lon', 'lat']].to_numpy()[labels == label]

            # A polygon needs at least 3 points
            if len(cluster_points) < 3:
                continue

            # Create a convex hull for the points in this cluster
            multi_point = MultiPoint(cluster_points)
            enclosing_polygon = multi_point.convex_hull

            # Add the new polygon to our list of GeoJSON features
            features.append(
                geojson.Feature(
                    geometry=enclosing_polygon,
                    properties={"cluster_id": int(label)}
                )
            )

        
        # Return GeoJSON feature collection
        if features:
            feature_collection = geojson.FeatureCollection(features)
            return feature_collection
        else:
            return None

    except Exception as e:
        print(f"Error in main function: {e}")
        return None
def predict_shark_habitat(geojson_file='sharks_zone2.geojson', points_per_polygon=200, 
                 prediction_date='2030-05-14', point_generation_method='adaptive',
                 epsilon=1, min_samples=5, shark_name=""):
    """
    Main function that performs the complete prediction pipeline
    Returns GeoJSON feature collection
    """
    try:
        # Load models
        models = load_models()
        # Load GeoJSON polygons
        sharks = json.load(open('sharks.json'))
        prediction_points = pd.read_parquet('prediction_points.parquet')
        shark=next((shark for shark in sharks if shark['common_name'] == shark_name), None)
        if shark:
            min_temp_c = shark['min_temp_c']
            max_temp_c = shark['max_temp_c']
            min_chl = shark['min_chl_a_mg_m3']
            max_chl = shark['max_chl_a_mg_m3']
            results = batch_predict(prediction_points, models)
            dframe = pd.DataFrame(results)
            
            filtered_coords = dframe.loc[(dframe['predicted_sst'].between(min_temp_c, max_temp_c)) & (dframe['predicted_chla'].between(min_chl, max_chl)),['lon', 'lat']].to_numpy()
        # Generate prediction points
        # prediction_points = generate_prediction_points(
        #     gdf,
        #     points_per_polygon=points_per_polygon,
        #     method=point_generation_method,
        #     target_date=prediction_date
        # )
        # Save generated prediction points to CSV
        # try:
        #     prediction_points.to_parquet('prediction_points.parquet', index=False)
        # except Exception:
        #     pass
        # Run batch predictions
        
            
            # Convert to DataFrame for clustering
            
            if len(filtered_coords) == 0:
                return None
            # Perform DBSCAN Clustering
            db_presence = DBSCAN(eps=epsilon, min_samples=min_samples).fit(filtered_coords)
            labels = db_presence.labels_

            # Get the unique cluster labels, ignoring noise (-1)
            unique_labels = set(labels)
            unique_labels.discard(-1)

            # Create a polygon for each cluster
            features = []
            for label in unique_labels:
                # Get all points belonging to the current cluster
                cluster_points = filtered_coords[labels == label]

                # A polygon needs at least 3 points
                if len(cluster_points) < 3:
                    continue

                # Create a convex hull for the points in this cluster
                multi_point = MultiPoint(cluster_points)
                enclosing_polygon = multi_point.convex_hull

                # Add the new polygon to our list of GeoJSON features
                features.append(
                    geojson.Feature(
                        geometry=enclosing_polygon,
                        properties={"cluster_id": int(label)}
                    )
                )
        # Return GeoJSON feature collection
            if features:
                feature_collection = geojson.FeatureCollection(features)
                return feature_collection
            else:
                return None

    except Exception as e:
        print(f"Error in main function: {e}")
        return None

# Main execution
if __name__ == "__main__":
    # Configuration
    GEOJSON_FILE = 'sharks_zone2.geojson'  # Updated path
    POINTS_PER_POLYGON = 200
    PREDICTION_DATE = '2030-05-14'
    POINT_GENERATION_METHOD = 'adaptive'
    EPSILON = 2
    MIN_SAMPLES = 5
    SHARK_NAME = "Whitespotted Bamboshark"
    # Call main function
    result = predict_shark_presence(
        geojson_file=GEOJSON_FILE,
        points_per_polygon=POINTS_PER_POLYGON,
        prediction_date=PREDICTION_DATE,
        point_generation_method=POINT_GENERATION_METHOD,
        epsilon=EPSILON,
        min_samples=MIN_SAMPLES
    )
    habitat_result = predict_shark_habitat(
        geojson_file=GEOJSON_FILE,
        points_per_polygon=POINTS_PER_POLYGON,
        prediction_date=PREDICTION_DATE,
        point_generation_method=POINT_GENERATION_METHOD,
        epsilon=EPSILON,
        min_samples=MIN_SAMPLES,
        shark_name=SHARK_NAME
    )