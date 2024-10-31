# Import necessary libraries
import pandas as pd
import time
from pyDatalog import pyDatalog  # Import pyDatalog for FOL reasoning
from collections import defaultdict

# Load GTFS Data with WSL-compatible paths
df_stops = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/GTFS/stops.txt', dtype={'stop_id': str})
df_routes = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/GTFS/routes.txt', dtype={'route_id': str})
df_stop_times = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/GTFS/stop_times.txt', dtype={'stop_id': str, 'trip_id': str})
df_trips = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/GTFS/trips.txt', dtype={'trip_id': str, 'route_id': str})

# Create route-to-stops mapping for efficient lookups
trip_to_route = df_trips.set_index('trip_id')['route_id'].to_dict()
routetostops = defaultdict(list)

# Populate route-to-stops dictionary using stop_times data
for trip_id, group in df_stop_times.groupby('trip_id'):
    route_id = trip_to_route.get(trip_id)
    if route_id:
        # Sort stops by sequence if 'stop_sequence' exists
        stops = group.sort_values(by='stop_sequence')['stop_id'].tolist() if 'stop_sequence' in group else group['stop_id'].tolist()
        routetostops[route_id].extend(stops)

# Ensure each route only has unique stops in order
for route_id, stops in routetostops.items():
    routetostops[route_id] = list(dict.fromkeys(stops))  # Remove duplicates while preserving order

# -------------------- Brute-Force Approach --------------------

def direct_route_brute_force(startstop, endstop):
    """
    Finds all routes that connect startstop to endstop directly (without interchanges).
    
    Parameters:
    startstop (str): Starting stop ID.
    endstop (str): Ending stop ID.
    
    Returns:
    list: List of route IDs that provide a direct connection from startstop to endstop.
    """
    direct_routes = []
    for route_id, stops in routetostops.items():
        if startstop in stops and endstop in stops:
            # Check if startstop appears before endstop
            if stops.index(startstop) < stops.index(endstop):
                direct_routes.append(route_id)
    return direct_routes

# Test Brute-Force Approach
start_time = time.time()
direct_routes_brute = direct_route_brute_force('0', '2')  # Use actual stop IDs to test
end_time = time.time()
print("Brute-Force Direct Routes:", direct_routes_brute)
print("Execution Time (Brute-Force):", end_time - start_time, "seconds")

# -------------------- FOL Library-Based Reasoning Approach --------------------

# Initialize pyDatalog for FOL approach
pyDatalog.clear()  # Clear previous terms if any

# Define terms for FOL-based reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, X, Y, Z, IndexX, IndexY')

# Populate facts for route-to-stop relationships
for route_id, stops in routetostops.items():
    for index, stop_id in enumerate(stops):
        +RouteHasStop(route_id, stop_id, index)

# Debug: Query a few facts from RouteHasStop to verify
print("Sample facts in RouteHasStop:")
print(RouteHasStop(Z, X, IndexX)[:10])  # Display the first 10 facts to verify data loading

# Define rule for DirectRoute using FOL
DirectRoute(X, Y) <= (RouteHasStop(Z, X, IndexX) & RouteHasStop(Z, Y, IndexY) & (IndexX < IndexY))

# Function to query DirectRoute using FOL
def direct_route_fol(startstop, endstop):
    # Query DirectRoute and capture the result
    result = DirectRoute(startstop, endstop).data
    print("DirectRoute Query Result:", result)  # Debug: print the query result
    
    # Check if result is empty or has the correct structure
    if not result:
        print(f"No direct route found between stops {startstop} and {endstop}")
        return []
    return [route_id[0] for route_id in result]  # Extract route_id safely

# Test FOL Approach
start_time = time.time()
direct_routes_fol = direct_route_fol('0', '2')  # Use actual stop IDs to test
end_time = time.time()
print("FOL Direct Routes:", direct_routes_fol)
print("Execution Time (FOL):", end_time - start_time, "seconds")

# -------------------- Comparison of Approaches --------------------

print("\nComparison of Approaches:")
print("Brute-Force Direct Routes:", direct_routes_brute)
print("FOL Direct Routes:", direct_routes_fol)
