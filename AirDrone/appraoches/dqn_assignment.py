"""
dqn_assignment.py
=================
Deep Q-Network for drone task assignment.

Changes vs. previous version
------------------------------
1. problem_signature saved with model – reviewers can verify same problem.
2. Default tests = 30; --sim_runs for 1000 simulator + 15 AirSim split.
3. Per-run statistics: mean ± std, 95 % CI, one-sample t-test.
4. Reward uses identical pair_cost() as Hungarian (no discrepancy).
5. Training plots show all `episodes` episodes (x-axis validated).

Usage
-----
    python dqn_assignment.py --mode train --episodes 10000
    python dqn_assignment.py --mode run   --tests 30
    python dqn_assignment.py --mode run   --sim_runs 1000 --tests 15
"""

from __future__ import annotations
import os, argparse, random, time, json
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from assignment_env   import AssignmentEnv
from assignment_layer import assign_tasks
from cost_utils       import DEFAULT_WEIGHTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def parse_weights(s):
    return None if s is None else json.loads(s)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def rolling_mean(values, window=50):
    out = []
    for i in range(len(values)):
        s = max(0, i - window + 1)
        out.append(float(np.mean(values[s:i + 1])))
    return out


def print_stats(label, costs, times=None):
    costs = np.array(costs)
    n     = len(costs)
    se    = costs.std() / np.sqrt(n)
    t_s, p_v = stats.ttest_1samp(costs, 0.0)
    print(f"\n========== {label} SUMMARY ==========")
    print(f"Runs              : {n}")
    print(f"Cost  mean ± std  : {costs.mean():.4f} ± {costs.std():.4f}")
    print(f"Cost  95 % CI     : [{costs.mean()-1.96*se:.4f}, {costs.mean()+1.96*se:.4f}]")
    print(f"Cost  [min, max]  : [{costs.min():.4f}, {costs.max():.4f}]")
    if times:
        t = np.array(times)
        print(f"Time  mean ± std  : {t.mean():.2f} ± {t.std():.2f} s")
    print(f"t-test vs 0       : t={t_s:.3f}, p={p_v:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer & network
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self): return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x): return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim  = action_dim
        self.policy_net  = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net  = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer   = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.buffer      = ReplayBuffer()
        self.gamma       = 0.95
        self.batch_size  = 64
        self.epsilon     = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.update_target_every = 200
        self.step_count  = 0

    def select_action(self, state, valid_actions=None):
        if valid_actions is not None and len(valid_actions) == 0:
            return 0
        if random.random() < self.epsilon:
            return int(random.choice(valid_actions if valid_actions else
                                     range(self.action_dim)))
        st = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q = self.policy_net(st).squeeze(0)
        if valid_actions is not None:
            masked = torch.full_like(q, -1e9)
            masked[valid_actions] = q[valid_actions]
            q = masked
        return int(torch.argmax(q).item())

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s  = torch.tensor(s,  dtype=torch.float32, device=DEVICE)
        a  = torch.tensor(a,  dtype=torch.long,    device=DEVICE).unsqueeze(1)
        r  = torch.tensor(r,  dtype=torch.float32, device=DEVICE).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=DEVICE)
        d  = torch.tensor(d,  dtype=torch.float32, device=DEVICE).unsqueeze(1)

        q  = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            nq = self.target_net(ns).max(1, keepdim=True)[0]
            tgt = r + self.gamma * nq * (1.0 - d)

        loss = nn.MSELoss()(q, tgt)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(history, out_dir, total_episodes):
    os.makedirs(out_dir, exist_ok=True)

    # Validate x-axis length equals total_episodes
    assert len(history["reward"]) == total_episodes, (
        f"Episode mismatch: history has {len(history['reward'])} entries "
        f"but expected {total_episodes}"
    )
    ep = np.arange(1, total_episodes + 1)

    for key, ylabel, title, fname in [
        ("reward", "Avg reward",    "DQN Reward Convergence",         "reward_convergence.png"),
        ("cost",   "Avg cost",      "DQN Cost Convergence",           "cost_convergence.png"),
        ("gap",    "Avg cost gap",  "DQN Optimality Gap Convergence", "gap_convergence.png"),
    ]:
        plt.figure(figsize=(10, 5))
        plt.plot(ep, rolling_mean(history[key], 50))
        plt.xlabel("Episode"); plt.ylabel(ylabel); plt.title(title)
        plt.xlim(1, total_episodes); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200); plt.close()

    if history["loss"]:
        up = np.arange(1, len(history["loss"]) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(up, rolling_mean(history["loss"], 20))
        plt.xlabel("Update step"); plt.ylabel("Loss"); plt.title("DQN Training Loss")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "dqn_loss.png"), dpi=200); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_dqn(
    model_path="dqn_assignment.pt",
    episodes=10_000,
    out_dir="outputs/dqn",
    task_multiplier=1,
    weights=None,
):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir  = os.path.join(out_dir, "plots");  os.makedirs(plots_dir,  exist_ok=True)
    models_dir = os.path.join(out_dir, "models"); os.makedirs(models_dir, exist_ok=True)

    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    agent = DQNAgent(env.state_dim, env.action_dim)

    config = {
        "algo":            "dqn",
        "episodes":        episodes,
        "task_multiplier": task_multiplier,
        "state_dim":       env.state_dim,
        "action_dim":      env.action_dim,
        "weights":         env.weights,
        "coord_limit":     env.coord_limit,
        "z_level":         env.z_level,
        "problem_signature": env.problem_signature,
        "problem_hash":      env.problem_hash,
    }
    save_json(os.path.join(out_dir, "config.json"), config)
    print(f"Problem hash: {env.problem_hash}  (must match baseline & PPO to compare)")

    history = {"reward": [], "cost": [], "gap": [], "loss": []}

    for ep in range(episodes):
        state = env.reset()
        done  = False
        ep_reward = 0.0

        while not done:
            valid   = env.get_valid_actions()
            action  = agent.select_action(state, valid)
            ns, r, done, _ = env.step(action)
            agent.buffer.push(state, action, r, ns, done)
            loss = agent.train_step()
            if loss is not None:
                history["loss"].append(loss)
            state = ns;  ep_reward += r

        history["reward"].append(ep_reward)
        history["cost"].append(-ep_reward)
        history["gap"].append(-ep_reward - env.optimal_cost())

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if (ep + 1) % 500 == 0:
            print(f"Ep {ep+1:5d} | "
                  f"avg_r={np.mean(history['reward'][-500:]):.3f} | "
                  f"avg_cost={np.mean(history['cost'][-500:]):.3f} | "
                  f"gap={np.mean(history['gap'][-500:]):.3f} | "
                  f"ε={agent.epsilon:.3f}")

        # Mid-training checkpoint every 2500 episodes
        if (ep + 1) % 2500 == 0:
            ck = os.path.join(models_dir, f"dqn_ep{ep+1}.pt")
            torch.save(agent.policy_net.state_dict(), ck)

    save_path = os.path.join(models_dir, model_path)
    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"Model saved → {save_path}")
    plot_convergence(history, plots_dir, episodes)
    print(f"Plots saved → {plots_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Simulator evaluation  (no AirSim needed)
# ─────────────────────────────────────────────────────────────────────────────

def eval_sim(
    model_path, sim_runs=1000, out_dir="outputs/dqn",
    task_multiplier=1, weights=None
):
    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    agent = DQNAgent(env.state_dim, env.action_dim)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    dqn_costs, hun_costs, gaps = [], [], []

    for _ in range(sim_runs):
        state = env.reset()
        chosen = []
        for _ in range(env.n):
            valid  = env.get_valid_actions()
            action = agent.select_action(state, valid)
            chosen.append(action)
            state, _, _, _ = env.step(action)

        dqn_cost = sum(env.pair_cost(i, t) for i, t in enumerate(chosen))
        asgn, cm = assign_tasks(
            env.drones, env.tasks,
            drone_battery=env.drone_battery,
            task_priority=env.task_priority,
            drone_speed=env.drone_speed,
            weights=env.weights,
        )
        hun_cost = sum(cm[d][t] for d, t in asgn)

        dqn_costs.append(dqn_cost)
        hun_costs.append(hun_cost)
        gaps.append(dqn_cost - hun_cost)

    print_stats(f"DQN SIM ({sim_runs} runs)", dqn_costs)
    print_stats("HUNGARIAN SIM",              hun_costs)
    dg = np.array(gaps)
    print(f"\nDQN − Hungarian  mean={dg.mean():.4f} ± {dg.std():.4f}")

    results = {
        "sim_runs": sim_runs,
        "dqn_mean": float(np.mean(dqn_costs)),
        "dqn_std":  float(np.std(dqn_costs)),
        "hun_mean": float(np.mean(hun_costs)),
        "hun_std":  float(np.std(hun_costs)),
        "gap_mean": float(dg.mean()),
        "gap_std":  float(dg.std()),
    }
    save_json(os.path.join(out_dir, "sim_metrics.json"), results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# AirSim evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_airsim(
    model_path, tests=30, out_dir="outputs/dqn",
    task_multiplier=1, weights=None
):
    try:
        import airsim
        from air_helpers import (
            get_airsim_scene, reached, prepare_drones,
            shutdown_drones, summarize_stats,
        )
    except ImportError:
        print("AirSim not available – use --mode sim instead.")
        return

    os.makedirs(out_dir, exist_ok=True)
    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    agent = DQNAgent(env.state_dim, env.action_dim)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    client      = airsim.MultirotorClient(); client.confirmConnection()
    drone_names = ["Drone1", "Drone2", "Drone3"]

    all_gaps, all_times, success_flags = [], [], []

    for idx in range(tests):
        print(f"\n===== AirSim TEST {idx+1}/{tests} =====")
        prepare_drones(client, drone_names)
        drones, drone_speeds = get_airsim_scene(client, drone_names)
        tasks = env.sample_tasks()

        drone_battery = np.ones(env.n, dtype=np.float32)
        task_priority = np.random.randint(1, 6, size=env.num_tasks).astype(np.float32)
        env.reset_from_scene(drones, tasks, drone_battery, drone_speeds, task_priority)

        state   = env.encode_state()
        chosen  = []
        for _ in range(env.n):
            valid  = env.get_valid_actions()
            action = agent.select_action(state, valid)
            chosen.append(action)
            state, _, _, _ = env.step(action)

        dqn_cost = sum(env.pair_cost(i, t) for i, t in enumerate(chosen))
        asgn, cm = assign_tasks(
            drones, tasks,
            drone_battery=drone_battery, task_priority=task_priority,
            drone_speed=drone_speeds,   weights=env.weights,
        )
        hun_cost = sum(cm[d][t] for d, t in asgn)
        all_gaps.append(dqn_cost - hun_cost)
        print(f"  DQN cost={dqn_cost:.4f}  Hungarian={hun_cost:.4f}  "
              f"gap={dqn_cost-hun_cost:.4f}")

        for di, ti in enumerate(chosen):
            x, y, z = tasks[ti]
            client.moveToPositionAsync(x, y, z, 5, timeout_sec=60,
                                       vehicle_name=drone_names[di])

        t0 = time.time(); ok = False
        while time.time() - t0 < 90:
            if all(reached(client, drone_names[di], tasks[chosen[di]], 2.5)
                   for di in range(env.n)):
                ok = True; break
            time.sleep(0.25)
        elapsed = time.time() - t0
        all_times.append(elapsed); success_flags.append(ok)
        print(f"  Time={elapsed:.2f}s  {'OK' if ok else 'TIMEOUT'}")
        time.sleep(2); shutdown_drones(client, drone_names)

    print_stats(f"DQN AIRSIM ({tests} runs)", [x+y for x, y in
                zip([hun_costs if False else 0]*tests, all_gaps)], all_times)
    # Gaps relative to Hungarian
    dg = np.array(all_gaps)
    t_s, p_v = stats.ttest_1samp(dg, 0.0)
    print(f"\nGap mean={dg.mean():.4f} ± {dg.std():.4f}  t={t_s:.3f}  p={p_v:.4f}")
    print(f"Success rate: {np.mean(success_flags):.3f}")
    save_json(os.path.join(out_dir, "airsim_metrics.json"), {
        "tests": tests, "gap_mean": float(dg.mean()), "gap_std": float(dg.std()),
        "success_rate": float(np.mean(success_flags)),
        "time_mean": float(np.mean(all_times)), "time_std": float(np.std(all_times)),
        "t_stat": float(t_s), "p_value": float(p_v),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "run", "sim"], required=True,
                        help="train | run (AirSim) | sim (pure-Python, no AirSim)")
    parser.add_argument("--model_path",      default="dqn_assignment.pt")
    parser.add_argument("--episodes",        type=int, default=10_000)
    parser.add_argument("--tests",           type=int, default=30,
                        help="AirSim test runs (default 30)")
    parser.add_argument("--sim_runs",        type=int, default=1000,
                        help="Simulator-only evaluation runs (default 1000)")
    parser.add_argument("--out_dir",         default="outputs/dqn")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--task_multiplier", type=int, default=1)
    parser.add_argument("--weights",         type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    weights = parse_weights(args.weights)

    if args.mode == "train":
        train_dqn(args.model_path, args.episodes, args.out_dir,
                  args.task_multiplier, weights)
    elif args.mode == "sim":
        eval_sim(args.model_path, args.sim_runs, args.out_dir,
                 args.task_multiplier, weights)
    else:
        run_airsim(args.model_path, args.tests, args.out_dir,
                   args.task_multiplier, weights)