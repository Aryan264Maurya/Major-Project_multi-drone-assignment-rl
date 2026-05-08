import os
import argparse
import random
import time

import airsim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def select_action(self, state, valid_actions=None, deterministic=False):
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(state_t)

            if valid_actions is not None and len(valid_actions) > 0:
                masked_logits = torch.full_like(logits, -1e9)
                masked_logits[:, valid_actions] = logits[:, valid_actions]
                logits = masked_logits

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
        values = torch.tensor(np.array(self.buffer.values), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(np.array(self.buffer.dones), dtype=torch.float32, device=DEVICE)

        returns = torch.zeros_like(rewards)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1.0 - dones[t])
            returns[t] = running_return

        advantages = returns - values
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

                logits, values_pred = self.model(mb_states)
                dist = torch.distributions.Categorical(logits=logits)

                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_pred, mb_returns)

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


def train_ppo(model_path="ppo_assignment.pt", episodes=10000, update_every=64, out_dir="outputs/ppo", task_multiplier=1):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    env = AssignmentEnv(n=3, task_multiplier=task_multiplier)
    state_dim = env.state_dim
    action_dim = env.action_dim

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
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            valid_actions = env.get_valid_actions()
            action, logprob, value = agent.select_action(state, valid_actions=valid_actions, deterministic=False)

            next_state, reward, done, _ = env.step(action)
            agent.buffer.add(state, action, logprob, reward, value, done)

            state = next_state
            episode_reward += reward

            if len(agent.buffer) >= update_every:
                metrics = agent.update()
                if metrics is not None:
                    history["loss"].append(
                        metrics["actor_loss"] + metrics["value_loss"] - metrics["entropy"]
                    )

        cost = -episode_reward
        optimal_cost = env.optimal_cost()
        gap = cost - optimal_cost

        history["reward"].append(episode_reward)
        history["cost"].append(cost)
        history["gap"].append(gap)

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

    save_path = os.path.join(models_dir, model_path)
    torch.save(agent.model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    plot_convergence(history, plots_dir)
    print(f"Training plots saved in {plots_dir}")


def run_airsim(model_path="ppo_assignment.pt", tests=5, out_dir="outputs/ppo", task_multiplier=1):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    env = AssignmentEnv(n=3, task_multiplier=task_multiplier)
    state_dim = env.state_dim
    action_dim = env.action_dim

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
            action, _, _ = agent.select_action(state, valid_actions=valid_actions, deterministic=True)
            chosen_tasks.append(action)
            state, _, done, _ = env.step(action)

        ppo_cost = 0.0
        for drone_idx, task_idx in enumerate(chosen_tasks):
            ppo_cost += env.pair_cost(drone_idx, task_idx)

        hungarian_assignments, cost_matrix = assign_tasks(
            drones,
            tasks,
            drone_battery=drone_battery,
            task_priority=task_priority,
            drone_speed=drone_speed,
        )
        hungarian_cost = sum(cost_matrix[d][t] for d, t in hungarian_assignments)

        print("PPO chosen tasks:", chosen_tasks)
        print("PPO total cost:", ppo_cost)
        print("Hungarian assignments:", hungarian_assignments)
        print("Hungarian total cost:", hungarian_cost)
        print("Difference:", abs(ppo_cost - hungarian_cost))

        if abs(ppo_cost - hungarian_cost) < 1e-3:
            total_matches += 1
        total_tests += 1
        all_gaps.append(ppo_cost - hungarian_cost)

        print("\nMoving drones using PPO assignment...")

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
            print("All drones reached PPO targets!")
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
    parser.add_argument("--model_path", default="ppo_assignment.pt")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--tests", type=int, default=5)
    parser.add_argument("--out_dir", default="outputs/ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--update_every", type=int, default=64)
    parser.add_argument("--task_multiplier", type=int, default=1)
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.mode == "train":
        train_ppo(
            model_path=args.model_path,
            episodes=args.episodes,
            update_every=args.update_every,
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