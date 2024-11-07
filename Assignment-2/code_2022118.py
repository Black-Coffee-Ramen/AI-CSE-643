# Boilerplate for AI Assignment — Knowledge Representation, Reasoning, and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict
import matplotlib.pyplot as plt  # Add this import at the top

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS files
df_stops = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/A2_Test_Boiler/GTFS/stops.txt', dtype={'stop_id': str})
df_routes = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/A2_Test_Boiler/GTFS/routes.txt', dtype={'route_id': str})
df_stop_times = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/A2_Test_Boiler/GTFS/stop_times.txt', dtype={'stop_id': str, 'trip_id': str})
df_trips = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/A2_Test_Boiler/GTFS/trips.txt', dtype={'trip_id': str, 'route_id': str})
df_fare_rules = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/A2_Test_Boiler/GTFS/fare_rules.txt', dtype={'route_id': str})

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    global route_to_stops, trip_to_route, stop_trip_count, merged_fare_df

    # Create trip_id to route_id mapping
    trip_to_route.update(df_trips.set_index('trip_id')['route_id'].to_dict())

    # Map route_id to a list of stops in order of their sequence
    for trip_id, group in df_stop_times.groupby('trip_id'):
        route_id = trip_to_route.get(trip_id)
        if route_id:
            stops = group.sort_values(by='stop_sequence')['stop_id'].tolist()
            route_to_stops[route_id].extend(stops)
    
    # Ensure each route only has unique stops in the order they appear
    for route_id in route_to_stops:
        route_to_stops[route_id] = list(dict.fromkeys(route_to_stops[route_id]))

    # Count trips per stop
    for stop_id in df_stop_times['stop_id']:
        stop_trip_count[stop_id] += 1

    df_fare_attributes = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/A2_Test_Boiler/GTFS/fare_attributes.txt', dtype={'fare_id': str})
    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on="fare_id", how="left")

# Top 5 busiest routes based on the number of trips
def get_busiest_routes():
    trip_counts = pd.Series(list(trip_to_route.values())).value_counts().head(5)
    return list(trip_counts.items())

# Top 5 stops with the most frequent trips
def get_most_frequent_stops():
    stop_counts = pd.Series(stop_trip_count).nlargest(5)
    return list(stop_counts.items())

# Top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    stop_route_counts = {stop: len(set([route for route, stops in route_to_stops.items() if stop in stops])) for stop in stop_trip_count}
    top_stops = pd.Series(stop_route_counts).nlargest(5)
    return list(top_stops.items())

# Top 5 pairs of stops with exactly one direct route between them
def get_stops_with_one_direct_route():
    direct_routes = []
    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            pair = (stops[i], stops[i + 1])
            if direct_routes.count(pair) == 1:
                direct_routes.append((pair, route_id))
    return direct_routes[:5]

def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph using Plotly
def visualize_stop_route_graph_interactive(route_to_stops):
    G = nx.Graph()
    for route, stops in route_to_stops.items():
        edges = [(stops[i], stops[i + 1]) for i in range(len(stops) - 1)]
        G.add_edges_from(edges)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
    plt.show()  # Added to enable visualization with matplotlib

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    direct_routes = []
    for route_id, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops:
            if stops.index(start_stop) < stops.index(end_stop):
                direct_routes.append(route_id)
    return direct_routes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, X, Y, Z, IndexX, IndexY')

def initialize_datalog():
    pyDatalog.clear()  # Clear previous terms
    for route_id, stops in route_to_stops.items():
        for index, stop_id in enumerate(stops):
            +RouteHasStop(route_id, stop_id, index)

# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # Clear any existing facts to avoid duplicate data
    pyDatalog.clear()

    # Ensure terms have been created for PyDatalog
    pyDatalog.create_terms('RouteHasStop, DirectRoute, X, Y, Z, IndexX, IndexY')

    # Populate the RouteHasStop fact with data
    for route_id, stops in route_to_stops.items():
        for index, stop_id in enumerate(stops):
            # Add each stop in the route to RouteHasStop with route ID, stop ID, and index in the route
            +RouteHasStop(route_id, stop_id, index)


# Query direct routes using PyDatalog
def query_direct_routes(start_stop, end_stop):
    DirectRoute(X, Y) <= (RouteHasStop(Z, X, IndexX) & RouteHasStop(Z, Y, IndexY) & (IndexX < IndexY))
    result = DirectRoute(start_stop, end_stop).data
    if result:
        return [route_id[0] for route_id in result if route_id]  # Safely unpack the first element of each tuple
    return []  # Return an empty list if result is empty



# Planning with Forward Chaining
def forward_chaining(start_stop_id, end_stop_id, via_stop, max_transfers):
    DirectRoute(X, Y) <= (RouteHasStop(Z, X, IndexX) & RouteHasStop(Z, Y, IndexY) & (IndexX < IndexY))
    route_to_via = DirectRoute(start_stop_id, via_stop).data
    route_from_via = DirectRoute(via_stop, end_stop_id).data
    result = [(route1, via_stop, route2) for route1 in route_to_via for route2 in route_from_via if len(set([route1, route2])) <= max_transfers]
    return result

# Planning with Backward Chaining
def backward_chaining(start_stop_id, end_stop_id, via_stop, max_transfers):
    DirectRoute(X, Y) <= (RouteHasStop(Z, X, IndexX) & RouteHasStop(Z, Y, IndexY) & (IndexX < IndexY))
    route_from_via = DirectRoute(end_stop_id, via_stop).data
    route_to_via = DirectRoute(via_stop, start_stop_id).data
    result = [(route1, via_stop, route2) for route1 in route_to_via for route2 in route_from_via if len(set([route1, route2])) <= max_transfers]
    return result


# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    direct_routes = query_direct_routes(start_stop_id, end_stop_id)
    if direct_routes:
        return [(route, start_stop_id, end_stop_id) for route in direct_routes]
    else:
        return forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers)

# Example usage
create_kb()  # Initialize knowledge base
initialize_datalog()

# Query examples
print("Busiest Routes:", get_busiest_routes())
print("Most Frequent Stops:", get_most_frequent_stops())
print("Top 5 Busiest Stops:", get_top_5_busiest_stops())
print("Direct Routes between stops:", direct_route_brute_force('1', '10'))
print("PDDL-style planning result:", pddl_planning('1', '10', '5', 1))

# Visualize the stop-route graph
visualize_stop_route_graph_interactive(route_to_stops)
