import os
import folium
import json
import pandas as pd
import networkx as nx
import osmnx as ox
import geopandas as gpd
from tqdm import tqdm

# Configure environment
os.environ["SHAPE_RESTORE_SHX"] = "YES"

# Configuration
ROAD_NETWORK_FILE = "road_network.graphml"
CLOSURE_FILE = "closure_nodes.json"
OUTPUT_MAP = "roundtrip_routes.html"
OUTPUT_EXCEL = "roundtrip_results.xlsx"


def load_data():
    print("Loading data...")
    road_network = ox.load_graphml(ROAD_NETWORK_FILE)

    with open(CLOSURE_FILE) as f:
        closures = json.load(f)

    tract_data = gpd.read_file(
        r'C:\Users\Muddasir\OneDrive\Desktop\sshkeys\tl_2018_17_tract_3\tl_2018_17_tract.shp')
    tract_data['GEOID'] = tract_data['GEOID'].astype(str)

    placekeys = pd.read_csv(
        r"C:\Users\Muddasir\OneDrive\Desktop\sshkeys\Project\data\2018_1_inverted_tractPlacekey_justChicagoBoth.csv")

    return road_network, closures, tract_data, placekeys


def get_centroid(tract_row):
    tract_mercator = tract_row.to_crs(epsg=3857)
    centroid = tract_mercator.geometry.centroid.iloc[0]
    return gpd.GeoSeries([centroid], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]


def calculate_route_metrics(graph, route):
    total_distance = 0
    total_duration = 0

    for u, v in zip(route[:-1], route[1:]):
        edge_data = graph.get_edge_data(u, v)[0]
        length = edge_data.get('length', 0)
        speed = edge_data.get('speed', 50)
        total_distance += length
        total_duration += (length / 1000) / (speed / 60)

    return total_distance, total_duration


def calculate_roundtrip(graph, start_point, waypoints, closures, max_attempts=10):
    working_graph = graph.copy()
    points = [start_point] + waypoints + [start_point]
    total_dist = 0
    total_time = 0
    all_routes = []

    # Remove closure edges
    for closure in closures:
        node = closure["node_id"]
        if node in working_graph:
            working_graph.remove_edges_from(list(working_graph.edges(node)))

    for i in range(len(points)-1):
        orig = points[i]
        dest = points[i+1]

        try:
            orig_node = ox.distance.nearest_nodes(
                working_graph, orig[1], orig[0])
            dest_node = ox.distance.nearest_nodes(
                working_graph, dest[1], dest[0])
            route = nx.shortest_path(
                working_graph, orig_node, dest_node, weight="length")

            leg_dist, leg_time = calculate_route_metrics(working_graph, route)
            total_dist += leg_dist
            total_time += leg_time
            all_routes.append(route)

        except nx.NetworkXNoPath:
            return None, None, None

    return total_dist, total_time, all_routes


def visualize_route(m, graph, routes, color):
    for i, route in enumerate(routes):
        coords = [(graph.nodes[node]["y"], graph.nodes[node]["x"])
                  for node in route]
        folium.PolyLine(
            coords,
            color=color,
            weight=3,
            opacity=0.7,
            popup=f"Leg {i+1}"
        ).add_to(m)


def main():
    road_network, closures, tract_data, placekeys = load_data()
    results = []

    # Initialize map
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)

    # Process GEOIDs
    for geoid in tqdm(tract_data['GEOID'].unique()[:5], desc="Processing GEOIDs"):
        tract_row = tract_data[tract_data['GEOID'] == geoid]
        if tract_row.empty:
            continue

        centroid = get_centroid(tract_row)
        start_point = (centroid.y, centroid.x)

        # Get placekey pairs
        for i in range(0, len(placekeys)-1, 2):
            pk1 = placekeys.iloc[i]
            pk2 = placekeys.iloc[i+1]
            waypoints = [
                (pk1["Latitude"], pk1["Longitude"]),
                (pk2["Latitude"], pk2["Longitude"])
            ]

            # Calculate routes
            orig_dist, orig_time, orig_routes = calculate_roundtrip(
                road_network, start_point, waypoints, []
            )

            alt_dist, alt_time, alt_routes = calculate_roundtrip(
                road_network, start_point, waypoints, closures
            )

            if orig_routes and alt_routes:
                # Add to results
                results.append({
                    "GEOID": geoid,
                    "Placekey1": pk1["Placekey"],
                    "Placekey2": pk2["Placekey"],
                    "Original_Distance_km": round(orig_dist/1000, 2),
                    "Original_Time_min": round(orig_time, 1),
                    "Alternate_Distance_km": round(alt_dist/1000, 2),
                    "Alternate_Time_min": round(alt_time, 1),
                    "Distance_Diff_km": round((alt_dist - orig_dist)/1000, 2),
                    "Time_Diff_min": round(alt_time - orig_time, 1)
                })

                # Visualize
                visualize_route(m, road_network, orig_routes, "blue")
                visualize_route(m, road_network, alt_routes, "red")

    # Save outputs
    m.save(OUTPUT_MAP)
    pd.DataFrame(results).to_excel(OUTPUT_EXCEL, index=False)
    print(f"Map saved to {OUTPUT_MAP}")
    print(f"Results saved to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
