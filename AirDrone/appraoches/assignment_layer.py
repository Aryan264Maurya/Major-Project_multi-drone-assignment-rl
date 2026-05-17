"""
assignment_layer.py
===================
Hungarian-algorithm task assignment.

Exports
-------
assign_tasks          – solve optimal assignment for one scene
run_weight_ablation   – one-at-a-time weight sensitivity (used by baseline)
run_weight_grid_search – exhaustive grid search
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from cost_utils import (
    build_cost_matrix,
    DEFAULT_WEIGHTS,
    run_weight_ablation,        # re-export so callers can import from here
    run_weight_grid_search,     # re-export
)


def assign_tasks(
    drones,
    tasks,
    drone_battery=None,
    task_priority=None,
    drone_speed=None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Solve the drone→task assignment problem with the Hungarian algorithm.

    Parameters
    ----------
    drones        : array-like (n, 3)
    tasks         : array-like (m, 3)
    drone_battery : array-like (n,)  or None
    task_priority : array-like (m,)  or None
    drone_speed   : array-like (n, 3) or None
    weights       : cost-weight dict (defaults to DEFAULT_WEIGHTS)

    Returns
    -------
    assignments  : list of (drone_idx, task_idx) pairs
    cost_matrix  : full (n × m) float32 cost matrix
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

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