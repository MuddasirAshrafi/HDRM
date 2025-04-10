import osmnx as ox
import networkx as nx
import folium
import geopandas as gpd
import pandas as pd
import json
import requests
import placekey as pk
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from geopy.distance import geodesic
from shapely.geometry import LineString, Point
import statsmodels.formula.api as smf
import statsmodels.api as sm  # Needed for families

# OSRM API Base URL
OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving"

# -------------------------------
# Helper Functions
# -------------------------------


def load_road_network(file_path="road_network.graphml"):
    print("Loading road network...")
    return ox.load_graphml(filepath=file_path)


def load_closure_nodes(file_path="closure_nodes.json"):
    print("Loading closure nodes...")
    with open(file_path, "r") as f:
        return json.load(f)


def calculate_route_metrics(graph, route):
    total_distance = 0  # in meters
    total_duration = 0  # in minutes
    for u, v in zip(route[:-1], route[1:]):
        edge_data = graph.get_edge_data(u, v, default=None)
        if edge_data:
            length = edge_data[0].get('length', 0)  # meters
            speed = edge_data[0].get('speed', 50)     # km/h default
            total_distance += length
            total_duration += (length / 1000) / (speed / 60)
    return total_distance, total_duration


def convert_to_minutes_and_seconds(fractional_minutes):
    whole_minutes = int(fractional_minutes)
    seconds = round((fractional_minutes - whole_minutes) * 60)
    return f"{whole_minutes} minutes and {seconds} seconds"


def get_osrm_distance(lat1, lon1, lat2, lon2):
    url = f"{OSRM_BASE_URL}/{lon1},{lat1};{lon2},{lat2}?overview=false"
    print(
        f"Querying OSRM for route: ({lat1:.6f}, {lon1:.6f}) -> ({lat2:.6f}, {lon2:.6f})")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            result = response.json()
            distance_m = result['routes'][0]['distance']
            return distance_m  # in meters
        else:
            print(f"Non-200 response: {response.status_code}")
            return float('nan')
    except Exception as e:
        print(f"OSRM error: {e}")
        return float('nan')


def compute_route_geometry(graph, point_a, point_b):
    try:
        orig_node = ox.distance.nearest_nodes(graph, point_a[1], point_a[0])
        dest_node = ox.distance.nearest_nodes(graph, point_b[1], point_b[0])
        route = nx.shortest_path(graph, orig_node, dest_node, weight="length")
        route_coords = [(graph.nodes[node]["y"], graph.nodes[node]["x"])
                        for node in route]
        return route, route_coords
    except Exception as e:
        print(f"Error computing route: {e}")
        return None, None


def route_affected_by_closure(route, closure_nodes):
    if route is None:
        return 0
    for closure in closure_nodes:
        node_id = closure.get("node_id")
        if node_id in route:
            return 1
    return 0

# -------------------------------
# Data Loading & Reshaping
# -------------------------------


csv_path = r"C:\Users\Muddasir\OneDrive\Desktop\sshkeys\Project\data\2018_1_inverted_tractPlacekey_justChicagoBoth.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
print("Original Data (first few rows):")
print(df.head())

# Identify Placekey columns (those containing '@')
placekey_cols = [col for col in df.columns if "@" in col]

# Melt the DataFrame using 'tract' as the identifier
long_df = df.melt(
    id_vars=['tract'],
    value_vars=placekey_cols,
    var_name='Placekey',
    value_name='Visits'
)
# Drop rows with zero visits
long_df = long_df[long_df['Visits'] > 0].reset_index(drop=True)
print("\nLong-format Data (first few rows):")
print(long_df.head())

# -------------------------------
# Load Tract Shapefile & Compute Centroids
# -------------------------------

shp_path = r"C:\Users\Muddasir\OneDrive\Desktop\sshkeys\tl_2018_17_tract_3\tl_2018_17_tract.shp"
tract_data = gpd.read_file(shp_path)
tract_data['GEOID'] = tract_data['GEOID'].astype(str).str.strip()

tract_centroids = {}
for geoid in tract_data['GEOID'].unique():
    tract_row = tract_data[tract_data['GEOID'] == geoid]
    if tract_row.empty:
        continue
    tract_proj = tract_row.to_crs(epsg=3857)
    centroid_proj = tract_proj.geometry.centroid.iloc[0]
    centroid_wgs84 = gpd.GeoSeries(
        [centroid_proj], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
    tract_centroids[geoid] = (centroid_wgs84.y, centroid_wgs84.x)

long_df['tract'] = long_df['tract'].astype(str).str.strip()
long_df['tract_lat'] = long_df['tract'].apply(
    lambda x: tract_centroids.get(x, (float('nan'), float('nan')))[0])
long_df['tract_lon'] = long_df['tract'].apply(
    lambda x: tract_centroids.get(x, (float('nan'), float('nan')))[1])

# -------------------------------
# Convert Placekeys to Store Coordinates
# -------------------------------

store_coords = []
for pk_val in long_df['Placekey'].unique():
    try:
        lat, lon = pk.placekey_to_geo(pk_val)
        store_coords.append(
            {"Placekey": pk_val, "Latitude": lat, "Longitude": lon})
    except Exception as e:
        print(f"Failed to convert Placekey {pk_val}: {e}")
store_df = pd.DataFrame(store_coords)
print("\nStore Coordinates (first few rows):")
print(store_df.head())

long_df['store_lat'], long_df['store_lon'] = zip(
    *long_df['Placekey'].apply(pk.placekey_to_geo))

# -------------------------------
# Limit to a Sample of 100 Routes for Testing
# -------------------------------

sample_df = long_df.head(500).copy()
print("\nWorking on a sample of 100 routes for testing.")

# -------------------------------
# Compute OSRM Driving Distances and Route Geometry
# -------------------------------

sample_df['driving_distance'] = sample_df.apply(
    lambda row: get_osrm_distance(
        row['tract_lat'], row['tract_lon'], row['store_lat'], row['store_lon']),
    axis=1
)
sample_df['log_driving_distance'] = np.log(sample_df['driving_distance'] + 1)
print("\nSample routes with computed driving distances (in meters):")
print(sample_df[['tract', 'Placekey', 'driving_distance',
      'log_driving_distance']].head())

road_network = load_road_network(
    r"C:\Users\Muddasir\OneDrive\Desktop\sshkeys\road_network.graphml")

route_list = []
for idx, row in sample_df.iterrows():
    pt_a = (row['tract_lat'], row['tract_lon'])
    pt_b = (row['store_lat'], row['store_lon'])
    route, _ = compute_route_geometry(road_network, pt_a, pt_b)
    route_list.append(route)
sample_df['route'] = route_list

closure_nodes = load_closure_nodes(
    r"C:\Users\Muddasir\OneDrive\Desktop\sshkeys\closure_nodes.json")
sample_df['closure_effect'] = sample_df['route'].apply(
    lambda r: route_affected_by_closure(r, closure_nodes))
print("\nClosure effect counts (0 = not affected, 1 = affected):")
print(sample_df['closure_effect'].value_counts())

# -------------------------------
# Save Route Metrics & Visits to Results
# -------------------------------

results = []
for idx, row in sample_df.iterrows():
    results.append({
        "GEOID": row['tract'],
        "Placekey": row['Placekey'],
        "Visits": row['Visits'],
        "Driving Distance (m)": row['driving_distance'],
        "Log Driving Distance": row['log_driving_distance'],
        "Closure Effect": row['closure_effect']
    })
results_df = pd.DataFrame(results)

# IMPORTANT: Rename the column "Closure Effect" so it matches our regression formula.
results_df.rename(columns={"Closure Effect": "closure_effect"}, inplace=True)

# -------------------------------
# Regression Analysis & Visualization
# -------------------------------

results_df['log_original_distance'] = np.log(
    results_df["Driving Distance (m)"] + 1)
print("\nRegression Data (first few rows):")
print(results_df.head())

model_formula = "Visits ~ log_original_distance + closure_effect"
poisson_model = smf.glm(formula=model_formula,
                        data=results_df, family=sm.families.Poisson()).fit()
print("\nPoisson Model Summary:")
print(poisson_model.summary())

# Visualization 1: Boxplot of Visits by closure_effect.
plt.figure(figsize=(8, 6))
results_df.boxplot(column="Visits", by="closure_effect")
plt.title("Visits by Road Closure Effect")
plt.suptitle("")
plt.xlabel("Closure Effect (0 = Not Affected, 1 = Affected)")
plt.ylabel("Visits")
plt.show()

# Visualization 2: Scatter plot of log_original_distance vs. Visits colored by closure_effect.
plt.figure(figsize=(8, 6))
color_map = results_df['closure_effect'].map({0: 'blue', 1: 'red'})
plt.scatter(results_df['log_original_distance'],
            results_df['Visits'], c=color_map)
plt.xlabel("Log Original Distance (m)")
plt.ylabel("Visits")
plt.title("Visits vs. Log Original Distance\n(Red = Affected by Closure)")
plt.show()

# -------------------------------
# Visualize Routes & Closures with Folium
# -------------------------------

m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
for geoid in sample_df['tract'].unique():
    tract_row = tract_data[tract_data['GEOID'] == geoid]
    if tract_row.empty:
        continue
    tract_row_4326 = tract_row.to_crs(epsg=4326)
    folium.GeoJson(
        tract_row_4326.geometry.iloc[0],
        style_function=lambda x: {'color': 'blue',
                                  'weight': 2, 'fillOpacity': 0.3},
        name=f"GEOID: {geoid}"
    ).add_to(m)
    centroid = tract_centroids.get(geoid, (None, None))
    if centroid[0] is not None:
        folium.Marker(
            location=centroid,
            popup=f"GEOID {geoid} Centroid",
            icon=folium.Icon(color="blue")
        ).add_to(m)

for idx, row in sample_df.iterrows():
    if not (pd.isna(row['tract_lat']) or pd.isna(row['store_lat'])):
        folium.Marker(
            [row['tract_lat'], row['tract_lon']],
            popup=f"Tract: {row['tract']}",
            icon=folium.Icon(color="blue")
        ).add_to(m)
        folium.Marker(
            [row['store_lat'], row['store_lon']],
            popup=f"Store: {row['Placekey']}",
            icon=folium.Icon(color="red")
        ).add_to(m)
        if row['route'] is not None:
            route_coords = [(road_network.nodes[node]["y"],
                             road_network.nodes[node]["x"]) for node in row['route']]
            folium.PolyLine(
                locations=route_coords,
                color="green" if row['closure_effect'] == 0 else "orange",
                weight=3,
                opacity=0.8,
                popup=f"Closure Effect: {row['closure_effect']}"
            ).add_to(m)

for closure in closure_nodes:
    node_id = closure.get("node_id")
    if node_id is None:
        continue
    try:
        lat = road_network.nodes[node_id]["y"]
        lon = road_network.nodes[node_id]["x"]
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color="black",
            fill=True,
            fill_color="black",
            fill_opacity=0.9,
            popup=f"Road Closure: Node {node_id}"
        ).add_to(m)
    except Exception as e:
        print(f"Error adding closure marker: {e}")

map_output = "routes_with_closures.html"
m.save(map_output)
print(f"\nFolium map saved as '{map_output}'.")

results_df.to_excel("sample_route_results.xlsx", index=False)
print("Results saved to 'sample_route_results.xlsx'.")
