from datetime import datetime
import osmnx as ox
import json
import requests

# OSRM API Base URL
OSRM_BASE_URL = "http://router.project-osrm.org/route/v1/driving"

# Function to preprocess and save the road network


def preprocess_road_network(north, south, east, west, output_file="road_network.graphml"):
    print("Downloading road network...")
    road_network = ox.graph_from_bbox(
        north, south, east, west, network_type="drive")
    print("Road network downloaded.")
    # Save the road network as a GraphML file
    ox.save_graphml(road_network, filepath=output_file)
    print(f"Road network saved to {output_file}.")

# Function to fetch and process road closures


def fetch_road_closures(api_url, output_file="road_closures.json"):
    print("Fetching road closures...")
    response = requests.get(api_url)
    if response.status_code == 200:
        closures = response.json()
        print(f"Fetched {len(closures)} road closures.")
        # Save the closures to a file
        with open(output_file, "w") as f:
            json.dump(closures, f)
        print(f"Road closures saved to {output_file}.")
    else:
        print(f"Error fetching road closures: {response.status_code}")
        return []

# Function to map road closures to nodes and save the mapping


def map_road_closures_to_nodes(road_network_file, road_closures_file, output_file="closure_nodes.json"):
    print("Mapping road closures to nearest nodes...")
    # Load the road network
    road_network = ox.load_graphml(filepath=road_network_file)
    # Load the road closures
    with open(road_closures_file, "r") as f:
        road_closures = json.load(f)

    closure_nodes = []
    for closure in road_closures:
        try:
            # Check if start and end dates are present
            start_date_str = closure.get("applicationstartdate")
            end_date_str = closure.get("applicationenddate")

            if start_date_str and end_date_str:
                # Convert to datetime objects
                start_date = datetime.fromisoformat(start_date_str)
                end_date = datetime.fromisoformat(end_date_str)

                # Calculate the duration in days
                duration_days = (end_date - start_date).days

                # Proceed only if duration is greater than 30 days
                if duration_days > 30 and "location" in closure and closure["location"]:
                    latitude = float(closure["location"]["latitude"])
                    longitude = float(closure["location"]["longitude"])
                    closure_node = ox.distance.nearest_nodes(
                        road_network, longitude, latitude)
                    # Include the node ID and original closure details
                    closure["node_id"] = closure_node
                    closure_nodes.append(closure)  # Add the filtered closure

        except Exception as e:
            print(f"Error processing closure: {e}")

    print(
        f"Mapped {len(closure_nodes)} closures to nodes with duration > 30 days.")
    # Save the filtered and mapped closures
    with open(output_file, "w") as f:
        json.dump(closure_nodes, f)
    print(f"Closure nodes saved to {output_file}.")


# Example bounding box for the Chicago area
north, south, east, west = 41.99, 41.64, -87.52, -87.94
padding = 0.01
north += padding
south -= padding
east += padding
west -= padding

# Run Preprocessing
road_network_file = "road_network.graphml"
road_closures_file = "road_closures.json"
closure_nodes_file = "closure_nodes.json"
road_closures_api_url = "https://data.cityofchicago.org/resource/jdis-5sry.json"

# Preprocess road network
preprocess_road_network(north, south, east, west, road_network_file)
# Fetch and process road closures
fetch_road_closures(road_closures_api_url, road_closures_file)
# Map road closures to nodes
map_road_closures_to_nodes(
    road_network_file, road_closures_file, closure_nodes_file)
