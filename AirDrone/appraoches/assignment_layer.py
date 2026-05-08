# import numpy as np
# from scipy.optimize import linear_sum_assignment

# def assign_tasks(drones, tasks):
#     drones = np.array(drones, dtype=np.float32)
#     tasks = np.array(tasks, dtype=np.float32)

#     cost_matrix = np.zeros((len(drones), len(tasks)), dtype=np.float32)

#     for i in range(len(drones)):
#         for j in range(len(tasks)):
#             cost_matrix[i][j] = np.linalg.norm(drones[i] - tasks[j])

#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#     assignments = list(zip(row_ind.tolist(), col_ind.tolist()))

#     return assignments, cost_matrix


import numpy as np
from scipy.optimize import linear_sum_assignment


def _as_float32(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float32)


def _cosine_penalty(v1, v2):
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_sim = float(np.dot(v1, v2) / (n1 * n2))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


def build_cost_matrix(
    drones,
    tasks,
    drone_battery=None,
    task_priority=None,
    drone_speed=None,
    weights=None,
):
    drones = np.asarray(drones, dtype=np.float32)
    tasks = np.asarray(tasks, dtype=np.float32)

    n_drones = len(drones)
    n_tasks = len(tasks)

    drone_battery = _as_float32(drone_battery)
    task_priority = _as_float32(task_priority)
    drone_speed = _as_float32(drone_speed)

    if weights is None:
        weights = {
            "dist": 1.0,
            "alt": 0.25,
            "battery": 0.75,
            "priority": 0.75,
            "speed": 0.15,
            "turn": 0.10,
        }

    cost_matrix = np.zeros((n_drones, n_tasks), dtype=np.float32)

    for i in range(n_drones):
        d = drones[i]

        battery_factor = 1.0
        if drone_battery is not None:
            b = float(drone_battery[i])
            battery_factor = 1.0 / max(b, 0.05)

        speed_factor = 0.0
        if drone_speed is not None:
            s = drone_speed[i]
            if np.ndim(s) == 0:
                speed_mag = float(s)
            else:
                speed_mag = float(np.linalg.norm(s))
            speed_factor = abs(speed_mag - 1.0)

        for j in range(n_tasks):
            t = tasks[j]

            dist = float(np.linalg.norm(d - t))
            alt = float(abs(d[2] - t[2]))

            priority_factor = 0.0
            if task_priority is not None:
                p = float(task_priority[j])
                priority_factor = 1.0 / max(p, 1.0)

            turn_factor = 0.0
            if drone_speed is not None:
                v = drone_speed[i]
                if np.ndim(v) == 0:
                    v = np.array([v, 0.0, 0.0], dtype=np.float32)
                direction = t - d
                turn_factor = _cosine_penalty(v, direction)

            cost = (
                weights["dist"] * dist
                + weights["alt"] * alt
                + weights["battery"] * battery_factor * dist
                + weights["priority"] * priority_factor
                + weights["speed"] * speed_factor
                + weights["turn"] * turn_factor
            )
            cost_matrix[i, j] = cost

    return cost_matrix


def assign_tasks(
    drones,
    tasks,
    drone_battery=None,
    task_priority=None,
    drone_speed=None,
    weights=None,
):
    cost_matrix = build_cost_matrix(
        drones=drones,
        tasks=tasks,
        drone_battery=drone_battery,
        task_priority=task_priority,
        drone_speed=drone_speed,
        weights=weights,
    )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = list(zip(row_ind.tolist(), col_ind.tolist()))
    return assignments, cost_matrix