import os
import argparse
import random
import time
from collections import deque

import airsim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from assignment_env import AssignmentEnv
from assignment_layer import assign_tasks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()

        self.gamma = 0.95
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.update_target_every = 200
        self.step_count = 0

    def select_action(self, state, valid_actions=None):
        if valid_actions is not None and len(valid_actions) == 0:
            return 0

        if random.random() < self.epsilon:
            if valid_actions is None:
                return random.randint(0, self.action_dim - 1)
            return int(random.choice(valid_actions))

        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0)

        if valid_actions is not None:
            masked = torch.full_like(q_values, -1e9)
            masked[valid_actions] = q_values[valid_actions]
            q_values = masked

        return int(torch.argmax(q_values).item())

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())


def rolling_mean(values, window=50):
    if len(values) == 0:
        return []
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start:i + 1])))
    return out


def plot_convergence(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ep = np.arange(1, len(history["reward"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(ep, rolling_mean(history["reward"], 50))
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.title("DQN Reward Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_convergence.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(ep, rolling_mean(history["cost"], 50))
    plt.xlabel("Episode")
    plt.ylabel("Average cost")
    plt.title("DQN Cost Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cost_convergence.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(ep, rolling_mean(history["gap"], 50))
    plt.xlabel("Episode")
    plt.ylabel("Average cost gap")
    plt.title("DQN Optimality Gap Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gap_convergence.png"), dpi=200)
    plt.close()

    if len(history["loss"]) > 0:
        up = np.arange(1, len(history["loss"]) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(up, rolling_mean(history["loss"], 20))
        plt.xlabel("Update")
        plt.ylabel("Loss")
        plt.title("DQN Training Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "dqn_loss.png"), dpi=200)
        plt.close()


def get_airsim_positions(client, drone_names):
    drones = []
    for name in drone_names:
        state = client.getMultirotorState(vehicle_name=name)
        pos = state.kinematics_estimated.position
        drones.append((pos.x_val, pos.y_val, pos.z_val))
    return drones


def reached(client, drone_name, target, tol=2.5):
    state = client.getMultirotorState(vehicle_name=drone_name)
    pos = state.kinematics_estimated.position
    dx = pos.x_val - target[0]
    dy = pos.y_val - target[1]
    dz = pos.z_val - target[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5 < tol


def wait_for_targets(client, drone_names, targets, max_wait=90, tol=2.5, poll_dt=0.25):
    start = time.time()
    while time.time() - start < max_wait:
        all_reached = True
        for i, name in enumerate(drone_names):
            if not reached(client, name, targets[i], tol=tol):
                all_reached = False
                break
        if all_reached:
            return True
        time.sleep(poll_dt)
    return False


def prepare_drones(client, drone_names):
    client.reset()
    time.sleep(1)

    for name in drone_names:
        client.enableApiControl(True, vehicle_name=name)
        client.armDisarm(True, vehicle_name=name)

    for name in drone_names:
        client.takeoffAsync(vehicle_name=name).join()

    time.sleep(2)

    offsets = [(0, 0), (5, 0), (-5, 0)]
    targets = [(offsets[i][0], offsets[i][1], -10) for i in range(len(drone_names))]

    for i, name in enumerate(drone_names):
        client.moveToPositionAsync(
            targets[i][0], targets[i][1], targets[i][2],
            5,
            timeout_sec=20,
            vehicle_name=name
        )

    wait_for_targets(client, drone_names, targets, max_wait=40, tol=2.5)
    time.sleep(1)


def shutdown_drones(client, drone_names):
    for name in drone_names:
        try:
            client.hoverAsync(vehicle_name=name).join()
        except Exception:
            pass

    for name in drone_names:
        try:
            client.landAsync(vehicle_name=name).join()
        except Exception:
            pass

    for name in drone_names:
        try:
            client.armDisarm(False, vehicle_name=name)
            client.enableApiControl(False, vehicle_name=name)
        except Exception:
            pass


def train_dqn(model_path="dqn_assignment.pt", episodes=10000, out_dir="outputs/dqn", task_multiplier=1):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    env = AssignmentEnv(n=3, task_multiplier=task_multiplier)
    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = Agent(state_dim, action_dim)
    history = {"reward": [], "cost": [], "gap": [], "loss": []}

    for ep in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions=valid_actions)

            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.train_step()

            state = next_state
            episode_reward += reward

            if loss is not None:
                history["loss"].append(loss)

        cost = -episode_reward
        optimal_cost = env.optimal_cost()
        gap = cost - optimal_cost

        history["reward"].append(episode_reward)
        history["cost"].append(cost)
        history["gap"].append(gap)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if (ep + 1) % 500 == 0:
            print(
                f"Episode {ep + 1:5d} | "
                f"avg_reward={np.mean(history['reward'][-500:]):.3f} | "
                f"avg_cost={np.mean(history['cost'][-500:]):.3f} | "
                f"avg_gap={np.mean(history['gap'][-500:]):.3f} | "
                f"epsilon={agent.epsilon:.3f}"
            )

    torch.save(agent.policy_net.state_dict(), os.path.join(models_dir, model_path))
    print(f"Model saved to {os.path.join(models_dir, model_path)}")

    plot_convergence(history, plots_dir)
    print(f"Plots saved to {plots_dir}")


def run_airsim(model_path="dqn_assignment.pt", tests=5, out_dir="outputs/dqn", task_multiplier=1):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    env = AssignmentEnv(n=3, task_multiplier=task_multiplier)
    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = Agent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    client = airsim.MultirotorClient()
    client.confirmConnection()

    drone_names = ["Drone1", "Drone2", "Drone3"]

    total_matches = 0
    total_tests = 0
    all_times = []
    all_gaps = []

    for test_idx in range(tests):
        print(f"\n========== TEST {test_idx + 1} ==========")

        prepare_drones(client, drone_names)
        drones = get_airsim_positions(client, drone_names)
        tasks = env.sample_tasks()

        drone_battery = np.ones(env.n, dtype=np.float32)
        drone_speed = np.ones(env.n, dtype=np.float32)
        task_priority = np.random.randint(1, 6, size=env.num_tasks).astype(np.float32)

        env.reset_from_scene(drones, tasks, drone_battery, drone_speed, task_priority)

        print("Drone Positions:", drones)
        print("Tasks:", tasks)

        state = env.encode_state()
        chosen_tasks = []

        for _ in range(env.n):
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions=valid_actions)
            chosen_tasks.append(action)
            state, _, done, _ = env.step(action)

        dqn_cost = 0.0
        for drone_idx, task_idx in enumerate(chosen_tasks):
            dqn_cost += env.pair_cost(drone_idx, task_idx)

        hungarian_assignments, cost_matrix = assign_tasks(
            drones,
            tasks,
            drone_battery=drone_battery,
            task_priority=task_priority,
            drone_speed=drone_speed,
        )
        hungarian_cost = sum(cost_matrix[d][t] for d, t in hungarian_assignments)

        print("DQN chosen tasks:", chosen_tasks)
        print("DQN total cost:", dqn_cost)
        print("Hungarian assignments:", hungarian_assignments)
        print("Hungarian total cost:", hungarian_cost)
        print("Difference:", abs(dqn_cost - hungarian_cost))

        if abs(dqn_cost - hungarian_cost) < 1e-3:
            total_matches += 1
        total_tests += 1
        all_gaps.append(dqn_cost - hungarian_cost)

        print("\nMoving drones using DQN assignment...")

        for drone_idx, task_idx in enumerate(chosen_tasks):
            drone_name = drone_names[drone_idx]
            x, y, z = tasks[task_idx]
            client.moveToPositionAsync(
                x, y, z,
                5,
                timeout_sec=60,
                vehicle_name=drone_name
            )

        start = time.time()
        ok = False
        while time.time() - start < 90:
            all_reached = True
            for drone_idx, task_idx in enumerate(chosen_tasks):
                drone_name = drone_names[drone_idx]
                if not reached(client, drone_name, tasks[task_idx], tol=2.5):
                    all_reached = False
                    break
            if all_reached:
                ok = True
                break
            time.sleep(0.25)

        completion_time = time.time() - start
        all_times.append(completion_time)
        print(f"Completion time: {completion_time:.2f} sec")

        if ok:
            print("All drones reached DQN targets!")
        else:
            print("Timeout: one or more drones did not reach targets.")

        time.sleep(2)
        shutdown_drones(client, drone_names)

    print("\n========== SUMMARY ==========")
    print(f"Exact cost match count: {total_matches}/{total_tests}")
    print(f"Exact match ratio: {total_matches / total_tests:.3f}")
    print(f"Average completion time: {np.mean(all_times):.2f} sec")
    print(f"Average cost gap vs Hungarian: {np.mean(all_gaps):.3f}")
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "run"], required=True)
    parser.add_argument("--model_path", default="dqn_assignment.pt")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--tests", type=int, default=5)
    parser.add_argument("--out_dir", default="outputs/dqn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_multiplier", type=int, default=1)
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.mode == "train":
        train_dqn(
            model_path=args.model_path,
            episodes=args.episodes,
            out_dir=args.out_dir,
            task_multiplier=args.task_multiplier,
        )
    else:
        run_airsim(
            model_path=args.model_path,
            tests=args.tests,
            out_dir=args.out_dir,
            task_multiplier=args.task_multiplier,
        )