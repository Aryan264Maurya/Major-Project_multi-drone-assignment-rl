import itertools
import random
import numpy as np

from assignment_layer import assign_tasks


class AssignmentEnv:
    def __init__(self, n=3, task_multiplier=1, coord_limit=15.0, z_level=-10.0):
        self.n = n
        self.task_multiplier = task_multiplier
        self.num_tasks = n * task_multiplier
        self.coord_limit = coord_limit
        self.z_level = z_level

        self.step_idx = 0
        self.drones = None
        self.tasks = None
        self.drone_battery = None
        self.drone_speed = None
        self.task_priority = None
        self.task_done = None

        self.action_dim = self.num_tasks
        self.state_dim = (6 * self.n) + (6 * self.num_tasks)

    def sample_scene(self):
        drones = []
        tasks = []

        for _ in range(self.n):
            x = random.uniform(-self.coord_limit, self.coord_limit)
            y = random.uniform(-self.coord_limit, self.coord_limit)
            drones.append((x, y, self.z_level))

        for _ in range(self.num_tasks):
            x = random.uniform(-self.coord_limit, self.coord_limit)
            y = random.uniform(-self.coord_limit, self.coord_limit)
            tasks.append((x, y, self.z_level))

        drone_battery = np.random.uniform(0.6, 1.0, size=self.n).astype(np.float32)
        drone_speed = np.random.uniform(0.2, 1.2, size=self.n).astype(np.float32)
        task_priority = np.random.randint(1, 6, size=self.num_tasks).astype(np.float32)

        return drones, tasks, drone_battery, drone_speed, task_priority

    def reset(self):
        drones, tasks, drone_battery, drone_speed, task_priority = self.sample_scene()
        return self.reset_from_scene(drones, tasks, drone_battery, drone_speed, task_priority)

    def reset_from_scene(self, drones, tasks, drone_battery=None, drone_speed=None, task_priority=None):
        self.drones = np.asarray(drones, dtype=np.float32)
        self.tasks = np.asarray(tasks, dtype=np.float32)
        self.drone_battery = np.asarray(drone_battery if drone_battery is not None else np.ones(self.n), dtype=np.float32)
        self.drone_speed = np.asarray(drone_speed if drone_speed is not None else np.ones(self.n), dtype=np.float32)
        self.task_priority = np.asarray(task_priority if task_priority is not None else np.ones(self.num_tasks), dtype=np.float32)

        self.task_done = np.zeros(self.num_tasks, dtype=np.float32)
        self.step_idx = 0
        return self.encode_state()

    def sample_tasks(self):
        tasks = []
        for _ in range(self.num_tasks):
            x = random.uniform(-self.coord_limit, self.coord_limit)
            y = random.uniform(-self.coord_limit, self.coord_limit)
            tasks.append((x, y, self.z_level))
        return tasks

    def current_drone_index(self):
        return self.step_idx

    def get_valid_actions(self):
        return np.where(self.task_done == 0)[0].tolist()

    def encode_state(self):
        drones = self.drones.copy()
        tasks = self.tasks.copy()

        drones[:, 0] /= self.coord_limit
        drones[:, 1] /= self.coord_limit
        drones[:, 2] /= abs(self.z_level)

        tasks[:, 0] /= self.coord_limit
        tasks[:, 1] /= self.coord_limit
        tasks[:, 2] /= abs(self.z_level)

        battery = self.drone_battery.reshape(-1, 1)
        speed = self.drone_speed.reshape(-1, 1)
        priority = (self.task_priority / 5.0).reshape(-1, 1)
        mask = self.task_done.reshape(-1, 1)

        current_one_hot = np.zeros((self.n,), dtype=np.float32)
        if self.step_idx < self.n:
            current_one_hot[self.step_idx] = 1.0

        current_drone = self.drones[self.step_idx] if self.step_idx < self.n else self.drones[-1]
        dist_row = np.linalg.norm(self.tasks - current_drone, axis=1, keepdims=True)
        dist_row /= self.coord_limit

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

    def pair_cost(self, drone_idx, task_idx):
        d = self.drones[drone_idx]
        t = self.tasks[task_idx]

        dist = float(np.linalg.norm(d - t))
        alt = float(abs(d[2] - t[2]))
        battery_factor = 1.0 / max(float(self.drone_battery[drone_idx]), 0.05)
        priority_factor = 1.0 / max(float(self.task_priority[task_idx]), 1.0)
        speed_factor = abs(float(self.drone_speed[drone_idx]) - 1.0)

        cost = (
            1.0 * dist
            + 0.25 * alt
            + 0.75 * battery_factor * dist
            + 0.75 * priority_factor
            + 0.15 * speed_factor
        )
        return float(cost)

    def optimal_cost(self):
        active_drones = self.drones[:self.n]
        active_tasks = self.tasks[:self.num_tasks]
        assignments, cost_matrix = assign_tasks(
            active_drones,
            active_tasks,
            drone_battery=self.drone_battery,
            task_priority=self.task_priority,
            drone_speed=self.drone_speed,
        )
        return float(sum(cost_matrix[d][t] for d, t in assignments))

    def step(self, action):
        if self.step_idx >= self.n:
            return self.encode_state(), 0.0, True, {}

        if self.task_done[action] == 1:
            reward = -50.0
        else:
            reward = -self.pair_cost(self.step_idx, action)
            self.task_done[action] = 1.0
            self.step_idx += 1

        done = self.step_idx >= self.n
        next_state = self.encode_state()
        info = {
            "current_drone": self.step_idx - 1 if self.step_idx > 0 else 0,
            "task_done": self.task_done.copy(),
        }
        return next_state, reward, done, info