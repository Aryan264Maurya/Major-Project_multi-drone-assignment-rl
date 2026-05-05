import numpy as np
from scipy.optimize import linear_sum_assignment

def assign_tasks(drones, tasks):
    drones = np.array(drones, dtype=np.float32)
    tasks = np.array(tasks, dtype=np.float32)

    cost_matrix = np.zeros((len(drones), len(tasks)), dtype=np.float32)

    for i in range(len(drones)):
        for j in range(len(tasks)):
            cost_matrix[i][j] = np.linalg.norm(drones[i] - tasks[j])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = list(zip(row_ind.tolist(), col_ind.tolist()))
    return assignments, cost_matrix