import numpy as np
import matplotlib.pyplot as plt
import random, os
from tqdm import tqdm
from roomba_class import Roomba
import csv

# ### Setup Environment ###
def seed_everything(seed: int):
    """Seed everything for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def is_obstacle(position):
    """Check if the position is outside the grid boundaries (acting as obstacles)."""
    x, y = position
    return x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT

def setup_environment(seed=111):
    """Setup function for grid and direction definitions."""
    global GRID_WIDTH, GRID_HEIGHT, HEADINGS, MOVEMENTS
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    HEADINGS = ['N', 'E', 'S', 'W']
    MOVEMENTS = {
        'N': (0, -1),
        'E': (1, 0),
        'S': (0, 1),
        'W': (-1, 0),
    }
    print("Environment setup complete with a grid of size {}x{}.".format(GRID_WIDTH, GRID_HEIGHT))
    seed_everything(seed)

# ### Simulate Roomba Movements ###
def simulate_roomba(T, movement_policy, sigma):
    """Simulate Roomba's movement for T time steps with noisy observations."""
    start_pos = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
    start_heading = random.choice(HEADINGS)
    roomba = Roomba(MOVEMENTS, HEADINGS, is_obstacle, start_pos, start_heading, movement_policy)

    true_positions, observations, headings = [], [], []
    for _ in tqdm(range(T), desc=f"Simulating Movement ({movement_policy})"):
        position = roomba.move()
        heading = roomba.heading
        true_positions.append(position)
        headings.append(heading)

        # Generate noisy observation
        noise = np.random.normal(0, sigma, 2)
        observed_position = (position[0] + noise[0], position[1] + noise[1])
        observations.append(observed_position)

    return true_positions, headings, observations

# ### HMM Definitions ###
def emission_probability(state, observation, sigma):
    """Calculate the emission probability using a Gaussian distribution."""
    true_x, true_y = state[0]
    obs_x, obs_y = observation
    prob = np.exp(-((obs_x - true_x)**2 + (obs_y - true_y)**2) / (2 * sigma**2))
    return prob

def transition_probability(prev_state, curr_state, movement_policy):
    """Calculate transition probability based on movement policy."""
    prev_pos, prev_heading = prev_state
    curr_pos, curr_heading = curr_state
    dx, dy = MOVEMENTS[prev_heading]

    # Straight Until Obstacle: Only move in the current direction
    if movement_policy == "straight_until_obstacle":
        expected_pos = (prev_pos[0] + dx, prev_pos[1] + dy)
        if expected_pos == curr_pos:
            return 1.0
        elif is_obstacle(expected_pos) and curr_pos != prev_pos:
            return 0.25  # Random heading change
        else:
            return 0.0

    # Random Walk: Equal probability for any valid move
    if movement_policy == "random_walk":
        valid_moves = sum(not is_obstacle((prev_pos[0] + MOVEMENTS[head][0], prev_pos[1] + MOVEMENTS[head][1])) for head in HEADINGS)
        if curr_pos == (prev_pos[0] + dx, prev_pos[1] + dy):
            return 1.0 / valid_moves
        else:
            return 0.0

    return 0.0

def viterbi(observations, start_state, movement_policy, states, sigma):
    """Perform the Viterbi algorithm to find the most likely sequence of states."""
    T = len(observations)
    n_states = len(states)

    # Initialize probabilities and backpointers
    dp = np.zeros((T, n_states))
    backpointer = np.zeros((T, n_states), dtype=int)

    # State lookup
    state_index = {state: i for i, state in enumerate(states)}

    # Initialization
    for i, state in enumerate(states):
        dp[0, i] = emission_probability(state, observations[0], sigma)

    # Recursion
    for t in range(1, T):
        for i, curr_state in enumerate(states):
            max_prob, max_idx = -np.inf, -1
            for j, prev_state in enumerate(states):
                prob = dp[t - 1, j] * transition_probability(prev_state, curr_state, movement_policy) * emission_probability(curr_state, observations[t], sigma)
                if prob > max_prob:
                    max_prob, max_idx = prob, j
            dp[t, i] = max_prob
            backpointer[t, i] = max_idx

    # Backtracking
    best_path = []
    best_last_state = np.argmax(dp[T - 1])
    best_path.append(states[best_last_state])

    for t in range(T - 1, 0, -1):
        best_last_state = backpointer[t, best_last_state]
        best_path.append(states[best_last_state])

    return best_path[::-1]

# ### Helper Functions ###
def save_to_csv(seed_values, estimated_paths, filename="estimated_paths.csv"):
    """Save the estimated paths and corresponding seed values to a CSV file."""
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Seed", "Estimated Path"])
        for seed, path in zip(seed_values, estimated_paths):
            path_str = "; ".join([f"({pos[0]}, {pos[1]})" for pos, _ in path])
            writer.writerow([seed, path_str])
    print(f"Results saved to {filename}.")

# ### Main Execution ###
if __name__ == "__main__":
    seed_values = [42, 123, 789]
    sigma = 1.0
    T = 50
    policies = ['random_walk', 'straight_until_obstacle']
    all_estimated_paths = []

    for seed in seed_values:
        setup_environment(seed)
        states = [((x, y), h) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT) for h in HEADINGS]
        for policy in policies:
            true_positions, headings, observations = simulate_roomba(T, policy, sigma)
            estimated_path = viterbi(observations, (true_positions[0], headings[0]), policy, states, sigma)
            all_estimated_paths.append(estimated_path)
            print(f"Seed: {seed}, Policy: {policy}, Estimated Path Length: {len(estimated_path)}")

    save_to_csv(seed_values, all_estimated_paths)
