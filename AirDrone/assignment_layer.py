import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise ImportError("scipy is required. Install it with: pip install scipy")


def euclidean_distance(p1, p2):
    """
    Returns Euclidean distance between two 3D points.
    p1 and p2 should be like (x, y, z)
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def build_cost_matrix(drones, tasks):
    """
    Builds a cost matrix where cost[i][j] is the distance
    between drone i and task j.
    """
    n = len(drones)
    m = len(tasks)
    cost = np.zeros((n, m), dtype=float)

    for i in range(n):
        for j in range(m):
            cost[i][j] = euclidean_distance(drones[i], tasks[j])

    return cost


def assign_tasks(drones, tasks):
    """
    Assigns tasks to drones using Hungarian Algorithm.

    Parameters:
        drones: list of (x, y, z)
        tasks:  list of (x, y, z)

    Returns:
        assignments: list of (drone_index, task_index)
        cost_matrix: numpy array
    """
    if len(drones) == 0 or len(tasks) == 0:
        return [], np.array([])

    cost_matrix = build_cost_matrix(drones, tasks)

    n = len(drones)
    m = len(tasks)

    # Hungarian algorithm works on square matrices best.
    # If matrix is rectangular, pad with very large values.
    size = max(n, m)
    padded = np.full((size, size), 1e9, dtype=float)
    padded[:n, :m] = cost_matrix

    row_ind, col_ind = linear_sum_assignment(padded)

    assignments = []
    for r, c in zip(row_ind, col_ind):
        # Keep only real drone-task matches, ignore dummy padded matches
        if r < n and c < m:
            assignments.append((r, c))

    return assignments, cost_matrix


def print_assignments(drones, tasks):
    """
    Helper function to print assignments nicely.
    """
    assignments, cost_matrix = assign_tasks(drones, tasks)

    print("Cost Matrix:")
    print(cost_matrix)

    print("\nAssignments:")
    for drone_idx, task_idx in assignments:
        print(f"Drone {drone_idx} -> Task {task_idx} | "
              f"Drone Pos: {drones[drone_idx]} | Task Pos: {tasks[task_idx]}")

    return assignments