# This runs but shows errors while running test cases

import numpy as np
import pickle

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def dfs_limited(adj_matrix, start_node, goal_node, limit, visited):
    if start_node == goal_node:
        return [start_node]
    if limit <= 0:
        return None

    visited[start_node] = True
    for neighbor, cost in enumerate(adj_matrix[start_node]):
        if cost > 0 and not visited[neighbor]:
            path = dfs_limited(adj_matrix, neighbor, goal_node, limit - 1, visited)
            if path:
                return [start_node] + path

    visited[start_node] = False
    return None

def get_ids_path(adj_matrix, start_node, goal_node):
    depth = 0
    n = len(adj_matrix)
    while depth <= n:  # setting upper bound to n for safety
        visited = [False] * n
        result = dfs_limited(adj_matrix, start_node, goal_node, depth, visited)
        if result:
            return result
        depth += 1
    return None



# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

from collections import deque

def bfs_step(queue, visited, parents, adj_matrix, direction):
    current_node = queue.popleft()
    
    for neighbor, cost in enumerate(adj_matrix[current_node]):
        if cost > 0 and not visited[neighbor]:
            visited[neighbor] = True
            parents[neighbor] = current_node
            queue.append(neighbor)

def construct_path(parents_u, parents_v, meeting_point):
    path_u = []
    path_v = []
    
    # Build the path from start to meeting point
    node = meeting_point
    while node is not None:
        path_u.append(node)
        node = parents_u[node]
    
    # Build the path from meeting point to goal
    node = meeting_point
    while node is not None:
        path_v.append(node)
        node = parents_v[node]
    
    return path_u[::-1] + path_v[1:]

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    if start_node == goal_node:
        return [start_node]
    
    n = len(adj_matrix)
    visited_u = [False] * n
    visited_v = [False] * n
    parents_u = [None] * n
    parents_v = [None] * n
    
    queue_u = deque([start_node])
    queue_v = deque([goal_node])
    visited_u[start_node] = True
    visited_v[goal_node] = True
    
    while queue_u and queue_v:
        if queue_u:
            bfs_step(queue_u, visited_u, parents_u, adj_matrix, "forward")
        if queue_v:
            bfs_step(queue_v, visited_v, parents_v, adj_matrix, "backward")
        
        # Check for overlap
        for i in range(n):
            if visited_u[i] and visited_v[i]:
                return construct_path(parents_u, parents_v, i)
    
    return None


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def euclidean_dist(node1, node2, node_attributes):
    x1, y1 = map(float, node_attributes[node1])
    x2, y2 = map(float, node_attributes[node2])
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    open_list = [(0, start_node)]
    heapq.heapify(open_list)
    came_from = {start_node: None}
    
    g_costs = {start_node: 0}
    f_costs = {start_node: euclidean_dist(start_node, goal_node, node_attributes)}
    
    closed_list = set()
    
    while open_list:
        current_f_cost, current_node = heapq.heappop(open_list)
        
        if current_node == goal_node:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            return path[::-1]
        
        closed_list.add(current_node)
        
        for neighbor, connected in enumerate(adj_matrix[current_node]):
            if connected and neighbor not in closed_list:
                tentative_g_cost = g_costs[current_node] + 1
                
                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = tentative_g_cost
                    f_cost = tentative_g_cost + euclidean_dist(neighbor, goal_node, node_attributes)
                    f_costs[neighbor] = f_cost
                    heapq.heappush(open_list, (f_cost, neighbor))
                    came_from[neighbor] = current_node
    
    return None




# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

import heapq
import math

def heuristic(node1, node2, node_attributes):
    x1, y1 = node_attributes[node1]
    x2, y2 = node_attributes[node2]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def reconstruct_path(came_from_start, came_from_goal, meeting_point):
    path_start = []
    current = meeting_point
    while current is not None:
        path_start.append(current)
        current = came_from_start[current]
    path_start.reverse()
    
    path_goal = []
    current = meeting_point
    while current is not None:
        path_goal.append(current)
        current = came_from_goal[current]
    path_goal.pop(0)  # Remove the duplicate meeting point
    
    return path_start + path_goal

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    if start_node == goal_node:
        return [start_node]
    
    forward_queue = [(0, start_node)]
    backward_queue = [(0, goal_node)]
    
    forward_cost = {start_node: 0}
    backward_cost = {goal_node: 0}
    
    came_from_start = {start_node: None}
    came_from_goal = {goal_node: None}
    
    forward_visited = set()
    backward_visited = set()
    
    meeting_point = None
    
    while forward_queue and backward_queue:
        if forward_queue:
            forward_current_cost, forward_current = heapq.heappop(forward_queue)
            if forward_current in backward_visited:
                meeting_point = forward_current
                break
            forward_visited.add(forward_current)
            
            for neighbor, connected in enumerate(adj_matrix[forward_current]):
                if connected:
                    new_cost = forward_cost[forward_current] + 1
                    if neighbor not in forward_cost or new_cost < forward_cost[neighbor]:
                        forward_cost[neighbor] = new_cost
                        priority = new_cost + heuristic(neighbor, goal_node, node_attributes)
                        heapq.heappush(forward_queue, (priority, neighbor))
                        came_from_start[neighbor] = forward_current
        
        if backward_queue:
            backward_current_cost, backward_current = heapq.heappop(backward_queue)
            if backward_current in forward_visited:
                meeting_point = backward_current
                break
            backward_visited.add(backward_current)
            
            for neighbor, connected in enumerate(adj_matrix[backward_current]):
                if connected:
                    new_cost = backward_cost[backward_current] + 1
                    if neighbor not in backward_cost or new_cost < backward_cost[neighbor]:
                        backward_cost[neighbor] = new_cost
                        priority = new_cost + heuristic(neighbor, start_node, node_attributes)
                        heapq.heappush(backward_queue, (priority, neighbor))
                        came_from_goal[neighbor] = backward_current
    
    if meeting_point is not None:
        return reconstruct_path(came_from_start, came_from_goal, meeting_point)
    
    return None



# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def find_bridges(adj_matrix):
    n = len(adj_matrix)
    ids = [-1] * n
    low = [0] * n
    bridges = []
    time = [0]  # Using a list to allow pass-by-reference
    
    def dfs(at, parent):
        ids[at] = low[at] = time[0]
        time[0] += 1
        
        for neighbor, cost in enumerate(adj_matrix[at]):
            if cost == 0:  # No edge
                continue
            if ids[neighbor] == -1:  # Not visited
                dfs(neighbor, at)
                low[at] = min(low[at], low[neighbor])
                if low[neighbor] > ids[at]:
                    bridges.append((at, neighbor))
            elif neighbor != parent:  # Update low-link value
                low[at] = min(low[at], ids[neighbor])
    
    for i in range(n):
        if ids[i] == -1:
            dfs(i, -1)
    
    return bridges

def bonus_problem(adj_matrix):
    return find_bridges(adj_matrix)



if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')