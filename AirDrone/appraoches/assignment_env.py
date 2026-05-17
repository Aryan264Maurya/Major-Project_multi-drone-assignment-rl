"""
assignment_env.py
=================
Gym-like environment for drone task assignment.

Key guarantee
-------------
The step() reward is  −pair_cost(...)  with the SAME weights and the SAME
pair_cost() function used by the Hungarian baseline.  This ensures a fair
comparison: both methods optimise the identical scalar objective.

Problem signature
-----------------
Every env instance exposes `problem_signature` – a dict of all
hyperparameters that define the problem class.  Saving this dict alongside
a trained model allows reviewers to verify that two models were trained on
the identical problem (same n, task_multiplier, coord_limit, z_level,
weights).  If any field differs, the models are NOT comparable.
"""

from __future__ import annotations
import hashlib
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from assignment_layer import assign_tasks
from cost_utils import pair_cost, DEFAULT_WEIGHTS


class AssignmentEnv:
    # ------------------------------------------------------------------ init
    def __init__(
        self,
        n: int = 3,
        task_multiplier: int = 1,
        coord_limit: float = 15.0,
        z_level: float = -10.0,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.n             = n
        self.task_multiplier = task_multiplier
        self.num_tasks     = n * task_multiplier
        self.coord_limit   = coord_limit
        self.z_level       = z_level
        self.weights       = DEFAULT_WEIGHTS if weights is None else dict(weights)

        # State / action dimensions
        # Per drone  : pos(3) + battery(1) + speed(3) + one_hot(1) = 8
        # Per task   : pos(3) + priority(1) + done(1) + dist(1)   = 6
        self.action_dim = self.num_tasks
        self.state_dim  = (8 * self.n) + (6 * self.num_tasks)

        # Runtime state (initialised in reset)
        self.step_idx      = 0
        self.drones        = None
        self.tasks         = None
        self.drone_battery = None
        self.drone_speed   = None
        self.task_priority = None
        self.task_done     = None

    # -------------------------------------------------------- problem identity
    @property
    def problem_signature(self) -> Dict:
        """
        A dict that uniquely identifies the problem class.

        Two models are comparable iff their problem_signatures are equal.
        Save this alongside model weights.
        """
        sig = {
            "n":               self.n,
            "task_multiplier": self.task_multiplier,
            "num_tasks":       self.num_tasks,
            "coord_limit":     self.coord_limit,
            "z_level":         self.z_level,
            "state_dim":       self.state_dim,
            "action_dim":      self.action_dim,
            "weights":         dict(sorted(self.weights.items())),
        }
        return sig

    @property
    def problem_hash(self) -> str:
        """SHA-256 of the problem signature (first 16 hex chars)."""
        blob = json.dumps(self.problem_signature, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    # --------------------------------------------------------------- sampling
    def sample_scene(self):
        drones = [
            (random.uniform(-self.coord_limit, self.coord_limit),
             random.uniform(-self.coord_limit, self.coord_limit),
             self.z_level)
            for _ in range(self.n)
        ]
        tasks = [
            (random.uniform(-self.coord_limit, self.coord_limit),
             random.uniform(-self.coord_limit, self.coord_limit),
             self.z_level)
            for _ in range(self.num_tasks)
        ]

        drone_battery = np.random.uniform(0.6, 1.0, size=self.n).astype(np.float32)

        theta = np.random.uniform(0.0, 2.0 * np.pi, size=self.n)
        mag   = np.random.uniform(0.2, 1.2, size=self.n)
        drone_speed = np.stack(
            [mag * np.cos(theta), mag * np.sin(theta), np.zeros(self.n)],
            axis=1,
        ).astype(np.float32)

        task_priority = np.random.randint(1, 6, size=self.num_tasks).astype(np.float32)
        return drones, tasks, drone_battery, drone_speed, task_priority

    def sample_tasks(self):
        return [
            (random.uniform(-self.coord_limit, self.coord_limit),
             random.uniform(-self.coord_limit, self.coord_limit),
             self.z_level)
            for _ in range(self.num_tasks)
        ]

    # ------------------------------------------------------------------ reset
    def reset(self):
        scene = self.sample_scene()
        return self.reset_from_scene(*scene)

    def reset_from_scene(
        self,
        drones,
        tasks,
        drone_battery=None,
        drone_speed=None,
        task_priority=None,
    ):
        self.drones        = np.asarray(drones, dtype=np.float32)
        self.tasks         = np.asarray(tasks,  dtype=np.float32)

        self.drone_battery = np.asarray(
            drone_battery if drone_battery is not None
            else np.ones(self.n, dtype=np.float32),
            dtype=np.float32,
        )

        if drone_speed is None:
            drone_speed = np.tile(
                np.array([1.0, 0.0, 0.0], dtype=np.float32), (self.n, 1)
            )
        self.drone_speed = np.asarray(drone_speed, dtype=np.float32)

        self.task_priority = np.asarray(
            task_priority if task_priority is not None
            else np.ones(self.num_tasks, dtype=np.float32),
            dtype=np.float32,
        )

        self.task_done = np.zeros(self.num_tasks, dtype=np.float32)
        self.step_idx  = 0
        return self.encode_state()

    # --------------------------------------------------------------- indexing
    def current_drone_index(self) -> int:
        return self.step_idx

    def get_valid_actions(self) -> List[int]:
        return np.where(self.task_done == 0)[0].tolist()

    # ------------------------------------------------------------ state codec
    def encode_state(self) -> np.ndarray:
        """
        Flat state vector (length = 8·n + 6·num_tasks):

          [drone_pos_norm(3·n) | battery(n) | speed(3·n) |
           task_pos_norm(3·m)  | priority(m)| done(m)    |
           current_one_hot(n)  | dist_to_tasks(m)]
        """
        drones = self.drones.copy()
        tasks  = self.tasks.copy()

        drones[:, 0] /= self.coord_limit
        drones[:, 1] /= self.coord_limit
        drones[:, 2] /= abs(self.z_level)

        tasks[:, 0] /= self.coord_limit
        tasks[:, 1] /= self.coord_limit
        tasks[:, 2] /= abs(self.z_level)

        battery  = self.drone_battery.reshape(-1, 1)
        speed    = self.drone_speed.reshape(-1, 3)
        priority = (self.task_priority / 5.0).reshape(-1, 1)
        mask     = self.task_done.reshape(-1, 1)

        current_one_hot = np.zeros((self.n,), dtype=np.float32)
        if self.step_idx < self.n:
            current_one_hot[self.step_idx] = 1.0

        current_drone = (
            self.drones[self.step_idx] if self.step_idx < self.n
            else self.drones[-1]
        )
        dist_row = np.linalg.norm(
            self.tasks - current_drone, axis=1, keepdims=True
        ) / self.coord_limit

        state = np.concatenate([
            drones.flatten(),
            battery.flatten(),
            speed.flatten(),
            tasks.flatten(),
            priority.flatten(),
            mask.flatten(),
            current_one_hot.flatten(),
            dist_row.flatten(),
        ]).astype(np.float32)

        return state

    # ----------------------------------------------------------- cost helpers
    def pair_cost(self, drone_idx: int, task_idx: int) -> float:
        """
        Scalar cost for assigning drone_idx → task_idx.

        Uses the SAME pair_cost() from cost_utils as the Hungarian baseline,
        with the SAME self.weights.  Reward = −pair_cost().
        """
        return pair_cost(
            self.drones[drone_idx],
            self.tasks[task_idx],
            drone_battery=self.drone_battery[drone_idx],
            task_priority=self.task_priority[task_idx],
            drone_speed=self.drone_speed[drone_idx],
            weights=self.weights,
        )

    def optimal_cost(self) -> float:
        """Hungarian-optimal cost for the current scene."""
        assignments, cost_matrix = assign_tasks(
            self.drones[:self.n],
            self.tasks[:self.num_tasks],
            drone_battery=self.drone_battery,
            task_priority=self.task_priority,
            drone_speed=self.drone_speed,
            weights=self.weights,
        )
        return float(sum(cost_matrix[d][t] for d, t in assignments))

    # ------------------------------------------------------------------ step
    def step(self, action: int):
        if self.step_idx >= self.n:
            return self.encode_state(), 0.0, True, {}

        if action < 0 or action >= self.num_tasks:
            return self.encode_state(), -100.0, self.step_idx >= self.n, {"invalid_action": True}

        if self.task_done[action] == 1:
            reward = -50.0
        else:
            # Reward = negative cost (identical formula to Hungarian baseline)
            reward = -self.pair_cost(self.step_idx, action)
            self.task_done[action] = 1.0
            self.step_idx += 1

        done       = self.step_idx >= self.n
        next_state = self.encode_state()
        info = {
            "current_drone": max(0, self.step_idx - 1),
            "task_done":     self.task_done.copy(),
        }
        return next_state, reward, done, info