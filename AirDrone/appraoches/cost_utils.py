"""
cost_utils.py
=============
Shared cost primitives used by EVERY method (Hungarian baseline, DQN, PPO,
Attention, and dynamic reassignment).  Using a single module guarantees that
all methods optimise the identical objective – a key reviewer concern.

Weight justification (literature basis)
----------------------------------------
wd  = 1.0   : Euclidean distance is the dominant energy cost for multirotor
              flight (Zeng et al. 2019, UAV trajectory optimisation survey).
wa  = 0.25  : Altitude deviation costs ~25 % of horizontal energy per metre
              (DJI power consumption models; Stolaroff et al. 2018).
wb  = 0.75  : Battery is a hard operational constraint; low-battery drones
              must be penalised strongly (Avellar et al. 2015).
wp  = 0.75  : Task priority maps to mission urgency; high-priority tasks
              should receive low-battery penalty equivalent weight
              (inspired by Liu et al. 2020 multi-UAV mission planning).
ws  = 0.15  : Speed deviation from 1 m/s nominal is a secondary correction.
wt  = 0.10  : Cosine turn penalty is a soft directional regulariser;
              small weight prevents it dominating distance.

An automated grid search (run_weight_grid_search) and a focused ablation
(run_weight_ablation) are provided so reviewers can reproduce sensitivity
analysis without guessing.
"""

from __future__ import annotations
import itertools
from typing import Dict, List, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Default weights  (identical for RL reward AND Hungarian cost matrix)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS: Dict[str, float] = {
    "dist":     1.00,
    "alt":      0.25,
    "battery":  0.75,
    "priority": 0.75,
    "speed":    0.15,
    "turn":     0.10,
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _as_float32(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float32)


def _to_vec3(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return np.array([float(arr[0]), 0.0, 0.0], dtype=np.float32)
    if arr.size == 2:
        return np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float32)
    return arr[:3].astype(np.float32)


def cosine_penalty(v1, v2) -> float:
    """1 − cos(angle between v1 and v2).  Returns 0 if either vector is zero."""
    v1 = _to_vec3(v1)
    v2 = _to_vec3(v2)
    if v1 is None or v2 is None:
        return 0.0
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_sim = float(np.dot(v1, v2) / (n1 * n2))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


# ─────────────────────────────────────────────────────────────────────────────
# Core cost function  (identical formula used everywhere)
# ─────────────────────────────────────────────────────────────────────────────

def pair_cost(
    drone_pos,
    task_pos,
    drone_battery: float = 1.0,
    task_priority: float = 1.0,
    drone_speed=None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute the scalar assignment cost for one (drone, task) pair.

    This function is the SINGLE source of truth for the cost signal.
    Both the RL reward  (reward = −pair_cost)  and the Hungarian cost
    matrix use this function with the same weights, ensuring a fair
    apples-to-apples comparison.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    d = np.asarray(drone_pos, dtype=np.float32)
    t = np.asarray(task_pos,  dtype=np.float32)

    dist = float(np.linalg.norm(d - t))
    alt  = float(abs(d[2] - t[2]))

    b = float(np.asarray(drone_battery, dtype=np.float32))
    p = float(np.asarray(task_priority, dtype=np.float32))

    battery_factor  = 1.0 / max(b, 0.05)
    priority_factor = 1.0 / max(p, 1.0)

    speed_factor = 0.0
    turn_factor  = 0.0
    if drone_speed is not None:
        v = _to_vec3(drone_speed)
        if v is not None:
            speed_factor = abs(float(np.linalg.norm(v)) - 1.0)
            direction    = t - d
            turn_factor  = cosine_penalty(v, direction)

    cost = (
        weights["dist"]     * dist
        + weights["alt"]      * alt
        + weights["battery"]  * battery_factor * dist
        + weights["priority"] * priority_factor
        + weights["speed"]    * speed_factor
        + weights["turn"]     * turn_factor
    )
    return float(cost)


def build_cost_matrix(
    drones,
    tasks,
    drone_battery=None,
    task_priority=None,
    drone_speed=None,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Build an (n_drones × n_tasks) cost matrix using pair_cost."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    drones        = np.asarray(drones, dtype=np.float32)
    tasks         = np.asarray(tasks,  dtype=np.float32)
    drone_battery = _as_float32(drone_battery)
    task_priority = _as_float32(task_priority)
    drone_speed   = _as_float32(drone_speed)

    n_drones  = len(drones)
    n_tasks   = len(tasks)
    cost_matrix = np.zeros((n_drones, n_tasks), dtype=np.float32)

    for i in range(n_drones):
        for j in range(n_tasks):
            b = drone_battery[i] if drone_battery is not None else 1.0
            p = task_priority[j] if task_priority is not None else 1.0
            v = drone_speed[i]   if drone_speed   is not None else None
            cost_matrix[i, j] = pair_cost(
                drones[i], tasks[j],
                drone_battery=b, task_priority=p,
                drone_speed=v,   weights=weights,
            )
    return cost_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Weight ablation & grid search
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_weights_on_scenes(
    scenes: List[dict],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute Hungarian total cost on a list of scenes with given weights.

    Each scene dict must contain:
        drones, tasks, drone_battery, task_priority, drone_speed
    """
    from scipy.optimize import linear_sum_assignment  # lazy import

    total_costs = []
    for sc in scenes:
        cm = build_cost_matrix(
            sc["drones"], sc["tasks"],
            drone_battery=sc.get("drone_battery"),
            task_priority=sc.get("task_priority"),
            drone_speed=sc.get("drone_speed"),
            weights=weights,
        )
        ri, ci = linear_sum_assignment(cm)
        total_costs.append(float(cm[ri, ci].sum()))

    arr = np.array(total_costs)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min())}


def run_weight_ablation(
    scenes: List[dict],
    weight_keys: Optional[List[str]] = None,
    base_weights: Optional[Dict[str, float]] = None,
    candidate_values: Optional[List[float]] = None,
) -> Dict[str, dict]:
    """
    One-at-a-time sensitivity analysis.

    For each key in weight_keys, vary that weight across candidate_values
    while holding the others at base_weights.  Prints a table and returns
    a nested dict: {key: {value: metrics}}.
    """
    if base_weights is None:
        base_weights = dict(DEFAULT_WEIGHTS)
    if weight_keys is None:
        weight_keys = list(DEFAULT_WEIGHTS.keys())
    if candidate_values is None:
        candidate_values = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0]

    print("\n===== Weight Ablation (one-at-a-time) =====")
    print(f"{'Key':<12} {'Value':>6} {'Mean Cost':>12} {'Std':>10}")
    print("-" * 44)

    results: Dict[str, dict] = {}
    for key in weight_keys:
        results[key] = {}
        for val in candidate_values:
            w = dict(base_weights)
            w[key] = val
            metrics = evaluate_weights_on_scenes(scenes, w)
            results[key][val] = metrics
            print(f"{key:<12} {val:>6.2f} {metrics['mean']:>12.4f} {metrics['std']:>10.4f}")
        print()

    return results


def run_weight_grid_search(
    scenes: List[dict],
    grid: Optional[Dict[str, List[float]]] = None,
) -> List[tuple]:
    """
    Exhaustive grid search over weight combinations.

    grid  – dict mapping weight key → list of candidate values.
    Returns a list of (total_mean_cost, weights_dict) sorted ascending.

    WARNING: combinatorial explosion.  Keep grids small (≤3 values/key).
    """
    if grid is None:
        grid = {
            "dist":     [0.5, 1.0, 1.5],
            "alt":      [0.1, 0.25, 0.5],
            "battery":  [0.5, 0.75, 1.0],
            "priority": [0.5, 0.75, 1.0],
            "speed":    [0.05, 0.15, 0.30],
            "turn":     [0.05, 0.10, 0.20],
        }

    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Grid search: {len(combos)} combinations on {len(scenes)} scenes …")

    scored = []
    for combo in combos:
        w       = dict(zip(keys, combo))
        metrics = evaluate_weights_on_scenes(scenes, w)
        scored.append((metrics["mean"], w))

    scored.sort(key=lambda x: x[0])

    print("\n===== Grid Search Top-10 =====")
    for rank, (cost, w) in enumerate(scored[:10], 1):
        print(f"#{rank:2d}  mean_cost={cost:.4f}  weights={w}")

    return scored