import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq

# Generate arena with obstacles (1) and free space (0), obstacle probability is 0.2 by default
def generate_arena(rows: int, cols: int, obstacle_prob: float = 0.2) -> np.ndarray:
    arena: np.ndarray = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            if np.random.rand() < obstacle_prob:
                arena[i, j] = 1  # Mark as obstacle
    return arena

# Heuristic function for A* is a Manhattan distance
def heuristic(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* search algorithm to find a path from start to goal in the arena
def a_star_search(arena: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    # Initialize A* search
    rows, cols = arena.shape
    alive_nodes: list[tuple[int, int]] = []
    # Set alive_nodes as a priority queue with the start node having f score of 0
    heapq.heappush(alive_nodes, (0, start))
    node_expand: dict[tuple[int, int], tuple[int, int]] = {}
    # Initialize g and f scores of the start node
    g_score: dict[tuple[int, int], int] = {start: 0}
    f_score: dict[tuple[int, int], int] = {start: heuristic(start, goal)}
    # Keep track of nodes in the alive set
    alive_nodes_hash: set[tuple[int, int]] = {start}

    # While there are nodes to expand, keep searching
    while alive_nodes:
        # Pop the node with the lowest f score, ignore the score of the node
        _, current = heapq.heappop(alive_nodes)
        # Remove the node from the alive set
        alive_nodes_hash.remove(current)

        # If the goal is reached, reconstruct the path
        if current == goal:
            path = []
            while current in node_expand:
                path.append(current)
                current = node_expand[current]
            path.append(start)
            path.reverse()
            return path

        # Generate neighbors of the current node (top,bottom,right,left) that are not obstacles  
        neighbors = [(current[0] + i, current[1] + j) for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        neighbors = [n for n in neighbors if 0 <= n[0] < rows and 0 <= n[1] < cols and arena[n[0], n[1]] == 0]

        # Update the g and f scores of the neighbors
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + 1
            # Check if the neighbor is not in the alive set or has a lower g score
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                node_expand[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in alive_nodes_hash:
                    heapq.heappush(alive_nodes, (f_score[neighbor], neighbor))
                    alive_nodes_hash.add(neighbor)

    return None

# Animate the path found by A* search
def animate_path(arena: np.ndarray, path: list[tuple[int, int]]):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(arena, cmap='Greys', origin='upper')

    robot, = ax.plot([], [], 'bo')  # Robot position
    trail, = ax.plot([], [], 'r-')  # Robot trail

    def init():
        robot.set_data([], [])
        trail.set_data([], [])
        return robot, trail

    def update(frame):
        x_data, y_data = trail.get_data()
        x_data = list(np.append(x_data, frame[1]))
        y_data = list(np.append(y_data, frame[0]))
        robot.set_data((frame[1], frame[0]))
        trail.set_data((x_data, y_data))
        return robot, trail


    ani = animation.FuncAnimation(
        fig, update, frames=path, init_func=init, blit=True, repeat=False)
    
    plt.title('Robot Arena with A* Algorithm Animation')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.grid(True)
    plt.show()

# Main program
while True:
    try:
        rows: int = int(input("Enter number of rows: "))
        cols: int = int(input("Enter number of columns: "))
        if rows <= 0 or cols <= 0:
            print("Number of rows and columns must be positive integers.")
            continue
        obstacle_prob: float = float(input("Enter obstacle probability (0-1): "))
        if not 0 <= obstacle_prob <= 1:
            print("Obstacle probability must be between 0 and 1.")
            continue
        start_row: int = int(input(f"Enter start row (0 to {rows-1}): "))
        start_col: int = int(input(f"Enter start column (0 to {cols-1}): "))
        goal_row: int = int(input(f"Enter goal row (0 to {rows-1}): "))
        goal_col: int = int(input(f"Enter goal column (0 to {cols-1}): "))
        if not (0 <= start_row < rows and 0 <= start_col < cols and 0 <= goal_row < rows and 0 <= goal_col < cols):
            print("Start and goal positions must be within the arena.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter a valid number.")

start: tuple[int, int] = (start_row, start_col)
goal: tuple[int, int] = (goal_row, goal_col)

arena: np.ndarray = generate_arena(rows, cols, obstacle_prob)
arena[start[0], start[1]] = 0
arena[goal[0], goal[1]] = 0

path: list[tuple[int, int]] = a_star_search(arena, start, goal)
if path:
    print("Shortest path found:", path)
    print("Path length:", len(path) - 1) # -1 because the path includes start and goal
    animate_path(arena, path)
else:
    print("No path found!")
