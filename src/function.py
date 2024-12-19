import osmnx as ox
import networkx as nx
import folium
import geopandas as gpd
import pandas as pd
import json
import requests
import placekey as pk
from datetime import datetime
# OSRM API Base URL
OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving"

# Load preprocessed data


def load_road_network(file_path="road_network.graphml"):
    print("Loading road network...")
    return ox.load_graphml(filepath=file_path)


def load_closure_nodes(file_path="closure_nodes.json"):
    print("Loading closure nodes...")
    with open(file_path, "r") as f:
        return json.load(f)


# Step 1: Load Census Tract Shapefile
tract_shapefile = "HDRM/tl_2018_17_tract.shp"
tract_data = gpd.read_file(tract_shapefile)
tract_data['GEOID'] = tract_data['GEOID'].astype(str)

# Step 2: Extract Placekeys and GEOIDs from CSV
file_path = "HDRM/2018_1_inverted_tractPlacekey_justChicagoBoth.csv"
data = pd.read_csv(file_path)
geoids = data['tract'].astype(str)
placekeys = [col for col in data.columns if '@' in col]

# Step 3: Convert Placekeys to Latitude/Longitude
placekey_coords = []
for placekey in placekeys:
    try:
        lat, lon = pk.placekey_to_geo(placekey)
        placekey_coords.append(
            {"Placekey": placekey, "Latitude": lat, "Longitude": lon})
    except Exception as e:
        print(f"Failed to convert Placekey {placekey}: {e}")
placekey_df = pd.DataFrame(placekey_coords)

# Load preprocessed road network and closure nodes
road_network = load_road_network("road_network.graphml")
closure_nodes = load_closure_nodes("closure_nodes.json")

# Initialize map
m = folium.Map(location=[placekey_df["Latitude"].mean(),
               placekey_df["Longitude"].mean()], zoom_start=13)

# Step 8: Process Each GEOID and Placekey
n = 1  # Number of pairs to iterate
for geoid in geoids[:n]:
    print(f"Processing GEOID: {geoid}")

    # Highlight GEOID area and centroid
    tract_row = tract_data[tract_data['GEOID'] == geoid]
    if tract_row.empty:
        print(f"GEOID {geoid} not found in shapefile. Skipping...")
        continue

    tract_row = tract_row.to_crs(epsg=3857)  # Reproject to Web Mercator
    centroid = tract_row.geometry.centroid.iloc[0]  # Calculate centroid
    centroid_coords = gpd.GeoSeries(
        [centroid], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
    centroid_lat, centroid_lon = centroid_coords.y, centroid_coords.x

    folium.GeoJson(
        tract_row.geometry.to_crs(epsg=4326).iloc[0],
        style_function=lambda x: {'color': 'blue',
                                  'weight': 2, 'fillOpacity': 0.3},
        name=f"GEOID: {geoid}"
    ).add_to(m)
    folium.Marker(
        location=(centroid_lat, centroid_lon),
        popup=f"GEOID {geoid} Centroid",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    # Loop through each placekey
    for _, row in placekey_df.iloc[:n].iterrows():
        point_a = (centroid_lat, centroid_lon)
        point_b = (row["Latitude"], row["Longitude"])

        folium.Marker(
            location=point_b,
            popup=f"Placekey: {row['Placekey']}",
            icon=folium.Icon(color="red")
        ).add_to(m)

        # Calculate Original Route
        try:
            orig_node = ox.distance.nearest_nodes(
                road_network, point_a[1], point_a[0])
            dest_node = ox.distance.nearest_nodes(
                road_network, point_b[1], point_b[0])

            original_route = nx.shortest_path(
                road_network, orig_node, dest_node, weight="length"
            )
            route_coords_original = [
                (road_network.nodes[node]["y"], road_network.nodes[node]["x"]) for node in original_route
            ]
            folium.PolyLine(
                route_coords_original,
                color="blue",
                weight=5,
                opacity=0.7,
                popup=f"Original Route for Placekey {row['Placekey']}"
            ).add_to(m)

            # Calculate distance and duration using OSRM
            coords = f"{point_a[1]},{point_a[0]};{point_b[1]},{point_b[0]}"
            response = requests.get(f"{OSRM_BASE_URL}/{coords}?overview=full")
            if response.status_code == 200:
                data = response.json()
                distance = data["routes"][0]["distance"]
                duration = data["routes"][0]["duration"]

                print(
                    f"Original Route: Distance = {distance:.2f} meters, Duration = {duration/60:.2f} minutes")

            else:
                print(f"Failed to fetch OSRM data: {response.status_code}")

        except Exception as e:
            print(f"Failed to compute original route: {e}")
            continue

        # Alternate route calculation
try:
    current_route = original_route
    max_attempts = 5
    attempts = 0

    while attempts < max_attempts:
        print(f"Attempt {attempts + 1}: Checking closures...")
        affected = False
        for closure_node in closure_nodes:
            node_id = closure_node.get("node_id")
            if node_id is None:
                continue

            if node_id in current_route:
                edges_to_remove = list(road_network.edges(node_id))
                for edge in edges_to_remove:
                    road_network.remove_edge(*edge)
                    affected = True
                print(f"Removed edges for closure node: {node_id}")

        if not affected:
            print("No further closures affecting the route.")
            break

        try:
            current_route = nx.shortest_path(
                road_network, orig_node, dest_node, weight="length"
            )
            print(
                f"Recalculated alternate route after {attempts + 1} attempts.")
        except nx.NetworkXNoPath:
            print("No alternate route found. Exiting loop.")
            current_route = None
            break

        attempts += 1

    if current_route:
        route_coords_alternate = [
            (road_network.nodes[node]["y"], road_network.nodes[node]["x"]) for node in current_route
        ]
        folium.PolyLine(
            route_coords_alternate,
            color="black",
            weight=5,
            opacity=0.7,
            popup=f"Alternate Route for Placekey {row['Placekey']}"
        ).add_to(m)

        # Calculate distance and duration for the alternate route
        # Format the full path of the alternate route for OSRM
        osrm_coords = ";".join(f"{lon},{lat}" for lat,
                               lon in route_coords_alternate)
        response = requests.get(
            f"{OSRM_BASE_URL}/{osrm_coords}?overview=false"
        )
        if response.status_code == 200:
            if "routes" in data and len(data["routes"]) > 0:
                alt_distance = data["routes"][0]["distance"]
                alt_duration = data["routes"][0]["duration"]
                print(
                    f"Alternate Route: Distance = {alt_distance:.2f} meters, Duration = {alt_duration/60:.2f} minutes")
            else:
                print("OSRM response did not contain valid route information.")
        else:
            print(
                f"Failed to fetch OSRM data for alternate route: {response.status_code}")
    else:
        print(
            f"No valid alternate route found for Placekey {row['Placekey']} after {attempts} attempts.")

except Exception as e:
    print(
        f"Error while processing alternate route for Placekey {row['Placekey']}: {e}")


# Step 9: Add Road Closures to Map
print("Adding road closures to map...")

for closure_node in closure_nodes:
    try:
        # Extract the node ID
        node_id = closure_node.get("node_id")
        if node_id is None:
            print(f"Node ID not found in closure_node: {closure_node}")
            continue

        # Get latitude and longitude of the node from the road network
        latitude = road_network.nodes[node_id]["y"]
        longitude = road_network.nodes[node_id]["x"]

        # Add a marker for the road closure
        folium.CircleMarker(
            location=(latitude, longitude),
            radius=10,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.9,
            popup=f"Road Closure: Node {node_id}",
        ).add_to(m)

    except KeyError as e:
        print(
            f"Error accessing node attributes for closure node {closure_node}: {e}")
    except Exception as e:
        print(f"Unexpected error for closure node {closure_node}: {e}")


# Step 10: Save Map
output_map_path = "routes_with_geoid_and_placekeys.html"
m.save(output_map_path)
print(f"Map saved as {output_map_path}.")
