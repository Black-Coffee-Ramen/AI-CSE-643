# Import necessary libraries
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict, Counter

# Load data
df_stops = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/GTFS/stops.txt', dtype={'stop_id': str})
df_routes = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/GTFS/routes.txt', dtype={'route_id': str})
df_stop_times = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/GTFS/stop_times.txt', dtype={'stop_id': str, 'trip_id': str})
df_trips = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/GTFS/trips.txt', dtype={'trip_id': str, 'route_id': str})
df_fare_rules = pd.read_csv('/mnt/c/Users/athiy/Downloads/AI Course 2024/Assignment-2/GTFS/fare_rules.txt', dtype={'route_id': str})

# Create a dictionary for trip_id to route_id mapping for faster lookup
trip_to_route = df_trips.set_index('trip_id')['route_id'].to_dict()

# Knowledge Base Creation
# Create routetostops dictionary
routetostops = defaultdict(list)
for trip_id, group in df_stop_times.groupby('trip_id'):
    route_id = trip_to_route.get(trip_id)
    if route_id:  # Ensure the trip has a corresponding route_id
        stop_ids = group['stop_id'].tolist()
        routetostops[route_id].extend(stop_ids)

# Create stoptripcount dictionary
stoptripcount = Counter(df_stop_times['stop_id'])

# Top 5 busiest routes based on the number of trips
route_trip_count = Counter(df_trips['route_id'])
top_5_busiest_routes = route_trip_count.most_common(5)

# Top 5 stops with the most frequent trips
top_5_stops_by_trip_count = stoptripcount.most_common(5)

# Top 5 busiest stops based on the number of routes passing through them
stops_routes = defaultdict(set)
for route_id, stops in routetostops.items():
    for stop_id in stops:
        stops_routes[stop_id].add(route_id)

top_5_stops_by_routes = sorted([(stop, len(routes)) for stop, routes in stops_routes.items()], 
                               key=lambda x: x[1], reverse=True)[:5]

# Top 5 pairs of stops connected by exactly one direct route, sorted by combined frequency
stop_pairs = defaultdict(int)
for route_id, stops in routetostops.items():
    for i in range(len(stops) - 1):
        start, end = stops[i], stops[i + 1]
        pair = tuple(sorted((start, end)))  # Sort to avoid duplicate pairs
        stop_pairs[pair] += 1

single_route_pairs = {pair: freq for pair, freq in stop_pairs.items() if freq == 1}
top_5_stop_pairs = sorted(single_route_pairs.items(), key=lambda x: x[1], reverse=True)[:5]

# Display results
print("Top 5 Busiest Routes (Route ID, Number of Trips):", top_5_busiest_routes)
print("Top 5 Stops with Most Frequent Trips (Stop ID, Trip Count):", top_5_stops_by_trip_count)
print("Top 5 Busiest Stops by Routes Passing Through (Stop ID, Route Count):", top_5_stops_by_routes)
print("Top 5 Stop Pairs Connected by One Direct Route (Stop Pair, Combined Frequency):", top_5_stop_pairs)

# Plotting Graph with Plotly
# Create graph data for route to stops
edges = []
for route_id, stops in routetostops.items():
    for i in range(len(stops) - 1):
        edges.append((stops[i], stops[i + 1]))

# Create a graph for visualization
fig = go.Figure()

# Add edges to the graph
for start, end in edges:
    fig.add_trace(go.Scatter(
        x=[df_stops[df_stops['stop_id'] == start]['stop_lat'].values[0], 
           df_stops[df_stops['stop_id'] == end]['stop_lat'].values[0]],
        y=[df_stops[df_stops['stop_id'] == start]['stop_lon'].values[0], 
           df_stops[df_stops['stop_id'] == end]['stop_lon'].values[0]],
        mode='lines',
        line=dict(width=1, color='blue'),
        opacity=0.5
    ))

# Add nodes (stops) to the graph
for _, row in df_stops.iterrows():
    fig.add_trace(go.Scatter(
        x=[row['stop_lat']],
        y=[row['stop_lon']],
        mode='markers+text',
        marker=dict(size=6, color='red'),
        text=row['stop_name'],
        textposition="top center"
    ))

# Set the layout for the plot
fig.update_layout(
    title="Bus Route Network",
    xaxis_title="Latitude",
    yaxis_title="Longitude",
    showlegend=False
)

# Show the plot
fig.show()
