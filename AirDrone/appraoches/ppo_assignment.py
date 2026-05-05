import os
import argparse
import itertools
import random
import time

from collections import deque

import airsim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from assignment_layer import assign_tasks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AssignmentEnv:
    def __init__(self, n=3, coord_limit=15.0, z_level=-10.0):
        self.n = n
        self.coord_limit = coord_limit
        self.z_level = z_level
        self.permutations = list(itertools.permutations(range(n)))
        self.state_dim = (self.n * 3) + (self.n * 3) + (self.n * self.n)

    def sample_scene(self):
        drones = []
        tasks = []

        for _ in range(self.n):
            x = random.uniform(-self.coord_limit, self.coord_limit)
            y = random.uniform(-self.coord_limit, self.coord_limit)
            drones.append((x, y, self.z_level))

        for _ in range(self.n):
            x = random.uniform(-self.coord_limit, self.coord_limit)
            y = random.uniform(-self.coord_limit, self.coord_limit)
            tasks.append((x, y, self.z_level))

        return drones, tasks

    def sample_tasks(self):
        tasks = []
        for _ in range(self.n):
            x = random.uniform(-self.coord_limit, self.coord_limit)
            y = random.uniform(-self.coord_limit, self.coord_limit)
            tasks.append((x, y, self.z_level))
        return tasks

    def encode_state(self, drones, tasks):
        drones = np.array(drones, dtype=np.float32).copy()
        tasks = np.array(tasks, dtype=np.float32).copy()

        drones[:, 0] /= self.coord_limit
        drones[:, 1] /= self.coord_limit
        tasks[:, 0] /= self.coord_limit
        tasks[:, 1] /= self.coord_limit

        drones[:, 2] /= abs(self.z_level)
        tasks[:, 2] /= abs(self.z_level)

        dist_matrix = np.zeros((self.n, self.n), dtype=np.float32)
        for i in range(self.n):
            for j in range(self.n):
                dist_matrix[i][j] = np.linalg.norm(drones[i] - tasks[j])

        dist_matrix /= self.coord_limit

        state = np.concatenate([
            drones.flatten(),
            tasks.flatten(),
            dist_matrix.flatten()
        ]).astype(np.float32)

        return state

    def total_cost_for_perm(self, drones, tasks, perm):
        cost = 0.0
        for drone_idx, task_idx in enumerate(perm):
            d = np.array(drones[drone_idx], dtype=np.float32)
            t = np.array(tasks[task_idx], dtype=np.float32)
            cost += np.linalg.norm(d - t)
        return float(cost)

    def reward(self, drones, tasks, action_idx):
        perm = self.permutations[action_idx]
        return -self.total_cost_for_perm(drones, tasks, perm)

    def step(self, drones, tasks, action_idx):
        state = self.encode_state(drones, tasks)
        reward = self.reward(drones, tasks, action_idx)
        next_state = state.copy()
        done = True
        return state, reward, next_state, done


class PPOBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, logprob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        update_epochs=10,
        minibatch_size=32,
    ):
        self.gamma = gamma
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.model = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = PPOBuffer()

    def select_action(self, state, deterministic=False):
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state_t)
            dist = torch.distributions.Categorical(logits=logits)

            if deterministic:
                action = torch.argmax(logits, dim=1)
            else:
                action = dist.sample()

            logprob = dist.log_prob(action)

        return int(action.item()), float(logprob.item()), float(value.item())

    def update(self):
        if len(self.buffer) == 0:
            return None

        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.long, device=DEVICE)
        old_logprobs = torch.tensor(np.array(self.buffer.logprobs), dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32, device=DEVICE)
        old_values = torch.tensor(np.array(self.buffer.values), dtype=torch.float32, device=DEVICE)

        returns = rewards.clone()
        advantages = returns - old_values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        idxs = np.arange(len(states))
        batch_size = min(self.minibatch_size, len(states))

        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                logits, values = self.model(mb_states)
                dist = torch.distributions.Categorical(logits=logits)

                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)

                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_actor_loss += float(actor_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                num_updates += 1

        self.buffer.clear()

        return {
            "actor_loss": total_actor_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
        }


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
    plt.title("PPO Reward Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_convergence.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(ep, rolling_mean(history["cost"], 50))
    plt.xlabel("Episode")
    plt.ylabel("Average cost")
    plt.title("PPO Cost Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cost_convergence.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(ep, rolling_mean(history["gap"], 50))
    plt.xlabel("Episode")
    plt.ylabel("Average cost gap")
    plt.title("PPO Optimality Gap Convergence")
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
        plt.title("PPO Training Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ppo_loss.png"), dpi=200)
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
    targets = []
    for i in range(len(drone_names)):
        targets.append((offsets[i][0], offsets[i][1], -10))

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


def move_and_record_trajectories(client, drone_names, tasks, perm, max_wait=90, tol=2.5, poll_dt=0.25):
    trajectories = {name: [] for name in drone_names}

    for name in drone_names:
        p = client.getMultirotorState(vehicle_name=name).kinematics_estimated.position
        trajectories[name].append((p.x_val, p.y_val, p.z_val))

    for drone_idx, task_idx in enumerate(perm):
        drone_name = drone_names[drone_idx]
        x, y, z = tasks[task_idx]
        client.moveToPositionAsync(
            x, y, z,
            5,
            timeout_sec=30,
            vehicle_name=drone_name
        )

    start = time.time()
    while time.time() - start < max_wait:
        all_reached = True
        for i, name in enumerate(drone_names):
            state = client.getMultirotorState(vehicle_name=name)
            pos = state.kinematics_estimated.position
            trajectories[name].append((pos.x_val, pos.y_val, pos.z_val))
            if not reached(client, name, tasks[perm[i]], tol=tol):
                all_reached = False

        if all_reached:
            break

        time.sleep(poll_dt)

    total_time = time.time() - start
    return trajectories, total_time, all_reached


def plot_trajectories(trajectories, drones_start, tasks, perm, save_path, title):
    plt.figure(figsize=(8, 8))

    tasks_np = np.array(tasks)
    plt.scatter(tasks_np[:, 0], tasks_np[:, 1], marker="*", s=180, label="Tasks")

    start_np = np.array(drones_start)
    plt.scatter(start_np[:, 0], start_np[:, 1], marker="o", s=80, label="Drone start")

    for drone_name, points in trajectories.items():
        pts = np.array(points)
        if len(pts) >= 2:
            plt.plot(pts[:, 0], pts[:, 1], linewidth=2, label=drone_name)

    for i, task_idx in enumerate(perm):
        s = drones_start[i]
        t = tasks[task_idx]
        plt.plot([s[0], t[0]], [s[1], t[1]], linestyle="--", linewidth=1)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train_ppo(model_path="ppo_assignment.pt", episodes=10000, update_every=64, out_dir="outputs/ppo"):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    env = AssignmentEnv(n=3)
    state_dim = env.state_dim
    action_dim = len(env.permutations)

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        update_epochs=10,
        minibatch_size=32,
    )

    history = {
        "reward": [],
        "cost": [],
        "gap": [],
        "loss": [],
    }

    for ep in range(episodes):
        drones, tasks = env.sample_scene()
        state = env.encode_state(drones, tasks)

        action, logprob, value = agent.select_action(state, deterministic=False)
        _, reward, _, done = env.step(drones, tasks, action)

        cost = -reward
        hungarian_assignments, cost_matrix = assign_tasks(drones, tasks)
        optimal_cost = sum(cost_matrix[d][t] for d, t in hungarian_assignments)
        gap = cost - optimal_cost

        agent.buffer.add(state, action, logprob, reward, value, done)

        history["reward"].append(reward)
        history["cost"].append(cost)
        history["gap"].append(gap)

        if len(agent.buffer) >= update_every:
            metrics = agent.update()
            if metrics is not None:
                history["loss"].append(
                    metrics["actor_loss"] + metrics["value_loss"] - metrics["entropy"]
                )

        if (ep + 1) % 500 == 0:
            print(
                f"Episode {ep + 1:5d} | "
                f"avg_reward={np.mean(history['reward'][-500:]):.3f} | "
                f"avg_cost={np.mean(history['cost'][-500:]):.3f} | "
                f"avg_gap={np.mean(history['gap'][-500:]):.3f}"
            )

    metrics = agent.update()
    if metrics is not None:
        history["loss"].append(metrics["actor_loss"] + metrics["value_loss"] - metrics["entropy"])

    save_path = os.path.join(models_dir, model_path)
    torch.save(agent.model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    plot_convergence(history, plots_dir)
    print(f"Training plots saved in {plots_dir}")


def run_airsim(model_path="ppo_assignment.pt", tests=5, out_dir="outputs/ppo"):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    env = AssignmentEnv(n=3)
    state_dim = env.state_dim
    action_dim = len(env.permutations)

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.model.eval()

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

        print("Drone Positions:", drones)
        print("Tasks:", tasks)

        state = env.encode_state(drones, tasks)
        action, _, _ = agent.select_action(state, deterministic=True)
        perm = env.permutations[action]

        ppo_cost = env.total_cost_for_perm(drones, tasks, perm)
        hungarian_assignments, cost_matrix = assign_tasks(drones, tasks)
        hungarian_cost = sum(cost_matrix[d][t] for d, t in hungarian_assignments)

        print("PPO chosen permutation:", perm)
        print("PPO total cost:", ppo_cost)
        print("Hungarian assignments:", hungarian_assignments)
        print("Hungarian total cost:", hungarian_cost)
        print("Difference:", abs(ppo_cost - hungarian_cost))

        if abs(ppo_cost - hungarian_cost) < 1e-3:
            total_matches += 1
        total_tests += 1
        all_gaps.append(ppo_cost - hungarian_cost)

        print("\nMoving drones using PPO assignment...")
        trajectories, completion_time, ok = move_and_record_trajectories(
            client=client,
            drone_names=drone_names,
            tasks=tasks,
            perm=perm,
            max_wait=90,
            tol=2.5,
            poll_dt=0.25
        )

        all_times.append(completion_time)
        print(f"Completion time: {completion_time:.2f} sec")

        traj_path = os.path.join(plots_dir, f"ppo_trajectory_test_{test_idx + 1}.png")
        plot_trajectories(
            trajectories=trajectories,
            drones_start=drones,
            tasks=tasks,
            perm=perm,
            save_path=traj_path,
            title=f"PPO Flight Trajectories - Test {test_idx + 1}"
        )
        print(f"Trajectory plot saved to {traj_path}")

        if ok:
            print("All drones reached PPO targets!")
        else:
            print("Timeout: one or more drones did not reach targets.")
            for name in drone_names:
                state = client.getMultirotorState(vehicle_name=name)
                pos = state.kinematics_estimated.position
                print(name, (pos.x_val, pos.y_val, pos.z_val))

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
    parser.add_argument("--model_path", default="ppo_assignment.pt")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--tests", type=int, default=5)
    parser.add_argument("--out_dir", default="outputs/ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--update_every", type=int, default=64)
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.mode == "train":
        train_ppo(
            model_path=args.model_path,
            episodes=args.episodes,
            update_every=args.update_every,
            out_dir=args.out_dir
        )
    else:
        run_airsim(
            model_path=args.model_path,
            tests=args.tests,
            out_dir=args.out_dir
        )