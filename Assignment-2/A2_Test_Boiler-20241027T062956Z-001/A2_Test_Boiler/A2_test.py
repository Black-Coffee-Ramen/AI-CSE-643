# Import necessary functions from code_changeRollNumber.py

from collections import defaultdict

route_to_stops = defaultdict(list)  # Maps route_id to an ordered list of stop_ids
trip_to_route = {}  # Maps trip_id to route_id
stop_trip_count = defaultdict(int)  # Maps stop_id to count of trips stopping there
fare_rules = {}  # Maps route_id to fare information

from code_changeRollNumber import (
    direct_route_brute_force,
    query_direct_routes,
    forward_chaining,
    backward_chaining,
    pddl_planning,
    bfs_route_planner_optimized,
    create_kb,  # Ensure the data is loaded for testing
    prune_data,
    initialize_datalog,
    get_merged_fare_df,
    compute_route_summary,
    get_busiest_routes,  # New functions for testing
    get_most_frequent_stops,
    get_top_5_busiest_stops,
    get_stops_with_one_direct_route
)

# Sample public test inputs with expected outputs explicitly defined
test_inputs = {
    "direct_route": [
        ((2573, 1177), [10001, 1117, 1407]),  # Input -> Expected output
        ((2001, 2005), [10001, 1151])
    ],

    "forward_chaining": [
        ((22540, 2573, 4686, 1), [(10153, 4686, 1407)]),
        ((951, 340, 300, 1), [(294, 300, 712), (10453, 300, 712), (1211, 300, 712), (1158, 300, 712), 
                              (37, 300, 712), (1571, 300, 712), (49, 300, 712), (387, 300, 712), 
                              (1206, 300, 712), (1038, 300, 712), (10433, 300, 712), (121, 300, 712)])
    ],
    "backward_chaining": [
        ((2573, 22540, 4686, 1), [(1407, 4686, 10153)]),
        ((340, 951, 300, 1), [(712, 300, 121), (712, 300, 1211), (712, 300, 37), (712, 300, 387),
                              (712, 300, 49), (712, 300, 10453), (712, 300, 1038), (712, 300, 10433),
                              (712, 300, 1571)])
    ],
    "pddl_planning": [
        ((22540, 2573, 4686, 1), [(10153, 4686, 1407)]),
        ((951, 340, 300, 1), [(294, 300, 712), (10453, 300, 712), (1211, 300, 712), (1158, 300, 712), 
                              (37, 300, 712), (1571, 300, 712), (49, 300, 712), (387, 300, 712), 
                              (1206, 300, 712), (1038, 300, 712), (10433, 300, 712), (121, 300, 712)])
    ],
    "bfs_route": [
        ((22540, 2573, 10, 3), [(10153, 4686), (1407, 2573)]),
        ((4012, 4013, 10, 3), [(10004, 4013)])
    ],

    ### NOTE: The below values are just dummy values, the actual values are might differ! 
    "busiest_routes": [
        [(123, 456), (789, 234), (567, 235), (3456, 897), (345, 345)]
    ],
    "most_frequent_stops": [
        [(456, 456), (234, 765), (234, 765), (234, 657765), (3252, 35634)]
    ],
    "busiest_stops": [
        [(432243, 14543), (454235, 2452), (2452, 2454), (78568, 24352), (42352, 24532)]
    ],
    "stops_with_one_direct_route": [
        [((24527, 676), 542), ((243535, 8768), 2456), ((43262, 564), 65437),
         ((256, 56), 245), ((266, 256), 78)]
    ]
}

def check_output(expected, actual):
    """Function to compare expected and actual outputs."""
    return set(expected) == set(actual)

def test_direct_route_brute_force():
    for (start_stop, end_stop), expected_output in test_inputs["direct_route"]:
        actual_output = direct_route_brute_force(start_stop, end_stop)
        print(f"Test direct_route_brute_force ({start_stop}, {end_stop}): ", 
              "Pass" if check_output(expected_output, actual_output) else f"Fail (Expected: {expected_output}, Got: {actual_output})")

def test_query_direct_routes():
    for (start_stop, end_stop), expected_output in test_inputs["direct_route"]:
        actual_output = query_direct_routes(start_stop, end_stop)
        print(f"Test query_direct_routes ({start_stop}, {end_stop}): ", 
              "Pass" if check_output(expected_output, actual_output) else f"Fail (Expected: {expected_output}, Got: {actual_output})")

def test_forward_chaining():
    for (start_stop, end_stop, via_stop, max_transfers), expected_output in test_inputs["forward_chaining"]:
        actual_output = forward_chaining(start_stop, end_stop, via_stop, max_transfers)
        print(f"Test forward_chaining ({start_stop}, {end_stop}, {via_stop}, {max_transfers}): ", 
              "Pass" if check_output(expected_output, actual_output) else f"Fail (Expected: {expected_output}, Got: {actual_output})")

def test_backward_chaining():
    for (end_stop, start_stop, via_stop, max_transfers), expected_output in test_inputs["backward_chaining"]:
        actual_output = backward_chaining(start_stop, end_stop, via_stop, max_transfers)
        print(f"Test backward_chaining ({start_stop}, {end_stop}, {via_stop}, {max_transfers}): ", 
              "Pass" if check_output(expected_output, actual_output) else f"Fail (Expected: {expected_output}, Got: {actual_output})")

def test_pddl_planning():
    for (start_stop, end_stop, via_stop, max_transfers), expected_output in test_inputs["pddl_planning"]:
        actual_output = pddl_planning(start_stop, end_stop, via_stop, max_transfers)
        print(f"Test pddl_planning ({start_stop}, {end_stop}, {via_stop}, {max_transfers}): ", 
              "Pass" if check_output(expected_output, actual_output) else f"Fail (Expected: {expected_output}, Got: {actual_output})")

def test_bfs_route_planner():
    for (start_stop, end_stop, initial_fare, max_transfers), expected_output in test_inputs["bfs_route"]:
        pruned_df = prune_data(merged_fare_df, initial_fare)
        route_summary = compute_route_summary(pruned_df)
        actual_output = bfs_route_planner_optimized(start_stop, end_stop, initial_fare, route_summary, max_transfers)
        print(f"Test bfs_route_planner_optimized ({start_stop}, {end_stop}, {initial_fare}, {max_transfers}): ", 
              "Pass" if check_output(expected_output, actual_output) else f"Fail (Expected: {expected_output}, Got: {actual_output})")

# New test functions for the additional queries

def test_get_busiest_routes():
    expected_output = test_inputs["busiest_routes"][0]
    actual_output = get_busiest_routes()
    print(f"Test get_busiest_routes: ", 
          "Pass" if check_output(expected_output, actual_output) else f"Fail (Expected: {expected_output}, Got: {actual_output})")

def test_get_most_frequent_stops():
    expected_output = test_inputs["most_frequent_stops"][0]
    actual_output = get_most_frequent_stops()
    print(f"Test get_most_frequent_stops: ", 
          "Pass" if check_output(expected_output, actual_output) else f"Fail (Expected: {expected_output}, Got: {actual_output})")

def test_get_top_5_busiest_stops():
    expected_output = test_inputs["busiest_stops"][0]
    actual_output = get_top_5_busiest_stops()
    print(f"Test get_top_5_busiest_stops: ", 
          "Pass" if check_output(expected_output, actual_output) else f"Fail (Expected: {expected_output}, Got: {actual_output})")

def test_get_stops_with_one_direct_route():
    expected_output = test_inputs["stops_with_one_direct_route"][0]
    actual_output = get_stops_with_one_direct_route()
    print(f"Test get_stops_with_one_direct_route: ", 
          "Pass" if check_output(expected_output, actual_output) else f"Fail (Expected: {expected_output}, Got: {actual_output})")

if __name__ == "__main__":
    create_kb()  # Ensure the data is loaded before testing
    merged_fare_df = get_merged_fare_df()  # Use the function to retrieve the DataFrame
    initialize_datalog()
    
    # Run all tests
    test_direct_route_brute_force()
    test_query_direct_routes()
    test_forward_chaining()
    test_backward_chaining()
    test_pddl_planning()
    test_bfs_route_planner()
    
    # Run additional tests for the new queries
    test_get_busiest_routes()
    test_get_most_frequent_stops()
    test_get_top_5_busiest_stops()
    test_get_stops_with_one_direct_route()