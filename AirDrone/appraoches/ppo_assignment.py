"""
ppo_assignment.py
=================
PPO agent for drone task assignment.

Changes vs. previous version
------------------------------
1. PPO update fires ONCE per episode (end-of-episode), not mid-episode.
   This prevents the ~4000-episode under-training bug: mid-episode clears
   the buffer, causing many episodes to contribute zero gradient updates.
2. Training plots validated to show exactly `episodes` data points.
3. problem_signature saved with model.
4. Default tests = 30.  --sim_runs for 1000-sim / 15-AirSim split.
5. Statistical output: mean ± std, 95 % CI, one-sample t-test.

Usage
-----
    python ppo_assignment.py --mode train --episodes 10000
    python ppo_assignment.py --mode run   --tests 30
    python ppo_assignment.py --mode run   --sim_runs 1000 --tests 15
"""

from __future__ import annotations
import os, argparse, random, time, json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from assignment_env   import AssignmentEnv
from assignment_layer import assign_tasks

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
    costs = np.array(costs); n = len(costs)
    se = costs.std() / np.sqrt(n)
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
# PPO components
# ─────────────────────────────────────────────────────────────────────────────

class PPOBuffer:
    def __init__(self): self.clear()

    def clear(self):
        self.states   = []
        self.actions  = []
        self.logprobs = []
        self.rewards  = []
        self.values   = []
        self.dones    = []

    def add(self, s, a, lp, r, v, d):
        self.states.append(s); self.actions.append(a); self.logprobs.append(lp)
        self.rewards.append(r); self.values.append(v); self.dones.append(d)

    def __len__(self): return len(self.states)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
        )
        self.actor  = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)


class PPOAgent:
    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, clip_range=0.2,
                 entropy_coef=0.01, value_coef=0.5,
                 update_epochs=10, minibatch_size=32):
        self.gamma         = gamma
        self.clip_range    = clip_range
        self.entropy_coef  = entropy_coef
        self.value_coef    = value_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.model     = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer    = PPOBuffer()

    def select_action(self, state, valid_actions=None, deterministic=False):
        st = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(st)
            if valid_actions is not None and len(valid_actions) > 0:
                ml = torch.full_like(logits, -1e9)
                ml[:, valid_actions] = logits[:, valid_actions]
                logits = ml
            dist   = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, 1) if deterministic else dist.sample()
            lp     = dist.log_prob(action)
        return int(action.item()), float(lp.item()), float(value.item())

    def update(self):
        if len(self.buffer) == 0:
            return None

        states    = torch.tensor(np.array(self.buffer.states),   dtype=torch.float32, device=DEVICE)
        actions   = torch.tensor(np.array(self.buffer.actions),  dtype=torch.long,    device=DEVICE)
        old_lp    = torch.tensor(np.array(self.buffer.logprobs), dtype=torch.float32, device=DEVICE)
        rewards   = torch.tensor(np.array(self.buffer.rewards),  dtype=torch.float32, device=DEVICE)
        values    = torch.tensor(np.array(self.buffer.values),   dtype=torch.float32, device=DEVICE)
        dones     = torch.tensor(np.array(self.buffer.dones),    dtype=torch.float32, device=DEVICE)

        # Monte-Carlo returns
        returns = torch.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running   = rewards[t] + self.gamma * running * (1.0 - dones[t])
            returns[t] = running

        advantages = returns - values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totals = dict(actor_loss=0.0, value_loss=0.0, entropy=0.0)
        n_upd  = 0
        idxs   = np.arange(len(states))
        bs     = min(self.minibatch_size, len(states))

        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(states), bs):
                mb = idxs[start:start + bs]
                logits, vp = self.model(states[mb])
                dist   = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(actions[mb])
                ent    = dist.entropy().mean()

                ratio  = torch.exp(new_lp - old_lp[mb])
                adv    = advantages[mb]
                actor_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * adv,
                ).mean()
                value_loss = F.mse_loss(vp, returns[mb])
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * ent

                self.optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                totals["actor_loss"] += float(actor_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"]    += float(ent.item())
                n_upd += 1

        self.buffer.clear()
        return {k: v / max(1, n_upd) for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(history, out_dir, total_episodes):
    os.makedirs(out_dir, exist_ok=True)

    assert len(history["reward"]) == total_episodes, (
        f"Plot x-axis mismatch: {len(history['reward'])} episodes recorded "
        f"but expected {total_episodes}."
    )
    ep = np.arange(1, total_episodes + 1)

    for key, ylabel, title, fname in [
        ("reward", "Avg reward",   "PPO Reward Convergence",         "reward_convergence.png"),
        ("cost",   "Avg cost",     "PPO Cost Convergence",           "cost_convergence.png"),
        ("gap",    "Avg cost gap", "PPO Optimality Gap Convergence", "gap_convergence.png"),
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
        plt.xlabel("PPO update"); plt.ylabel("Combined loss"); plt.title("PPO Training Loss")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ppo_loss.png"), dpi=200); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(
    model_path="ppo_assignment.pt",
    episodes=10_000,
    out_dir="outputs/ppo",
    task_multiplier=1,
    weights=None,
):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir  = os.path.join(out_dir, "plots");  os.makedirs(plots_dir,  exist_ok=True)
    models_dir = os.path.join(out_dir, "models"); os.makedirs(models_dir, exist_ok=True)

    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    config = {
        "algo":              "ppo",
        "episodes":          episodes,
        "task_multiplier":   task_multiplier,
        "state_dim":         env.state_dim,
        "action_dim":        env.action_dim,
        "weights":           env.weights,
        "coord_limit":       env.coord_limit,
        "z_level":           env.z_level,
        "problem_signature": env.problem_signature,
        "problem_hash":      env.problem_hash,
    }
    save_json(os.path.join(out_dir, "config.json"), config)
    print(f"Problem hash: {env.problem_hash}  "
          f"(must match DQN/Hungarian configs to compare models)")

    history = {"reward": [], "cost": [], "gap": [], "loss": []}

    for ep in range(episodes):
        state     = env.reset()
        done      = False
        ep_reward = 0.0

        # Collect one full episode into buffer
        while not done:
            valid   = env.get_valid_actions()
            action, lp, v = agent.select_action(state, valid_actions=valid, deterministic=False)
            ns, r, done, _ = env.step(action)
            agent.buffer.add(state, action, lp, r, v, done)
            state = ns;  ep_reward += r

        # Single PPO update at end of episode (fixes under-training bug)
        metrics = agent.update()
        if metrics is not None:
            history["loss"].append(
                metrics["actor_loss"] + metrics["value_loss"] - metrics["entropy"]
            )

        history["reward"].append(ep_reward)
        history["cost"].append(-ep_reward)
        history["gap"].append(-ep_reward - env.optimal_cost())

        if (ep + 1) % 500 == 0:
            print(f"Ep {ep+1:5d} | "
                  f"avg_r={np.mean(history['reward'][-500:]):.3f} | "
                  f"avg_cost={np.mean(history['cost'][-500:]):.3f} | "
                  f"gap={np.mean(history['gap'][-500:]):.3f}")

        # Checkpoint every 2500 episodes
        if (ep + 1) % 2500 == 0:
            ck = os.path.join(models_dir, f"ppo_ep{ep+1}.pt")
            torch.save(agent.model.state_dict(), ck)
            print(f"  Checkpoint saved → {ck}")

    save_path = os.path.join(models_dir, model_path)
    torch.save(agent.model.state_dict(), save_path)
    print(f"Model saved → {save_path}")
    plot_convergence(history, plots_dir, episodes)
    print(f"Plots saved → {plots_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Simulator evaluation  (no AirSim)
# ─────────────────────────────────────────────────────────────────────────────

def eval_sim(
    model_path, sim_runs=1000, out_dir="outputs/ppo",
    task_multiplier=1, weights=None
):
    os.makedirs(out_dir, exist_ok=True)
    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.model.eval()

    ppo_costs, hun_costs, gaps = [], [], []

    for _ in range(sim_runs):
        state   = env.reset()
        chosen  = []
        for _ in range(env.n):
            valid   = env.get_valid_actions()
            action, _, _ = agent.select_action(state, valid, deterministic=True)
            chosen.append(action)
            state, _, _, _ = env.step(action)

        ppo_cost = sum(env.pair_cost(i, t) for i, t in enumerate(chosen))
        asgn, cm = assign_tasks(
            env.drones, env.tasks,
            drone_battery=env.drone_battery,
            task_priority=env.task_priority,
            drone_speed=env.drone_speed,
            weights=env.weights,
        )
        hun_cost = sum(cm[d][t] for d, t in asgn)

        ppo_costs.append(ppo_cost)
        hun_costs.append(hun_cost)
        gaps.append(ppo_cost - hun_cost)

    print_stats(f"PPO SIM ({sim_runs} runs)", ppo_costs)
    print_stats("HUNGARIAN SIM",              hun_costs)
    dg = np.array(gaps)
    print(f"\nPPO − Hungarian  mean={dg.mean():.4f} ± {dg.std():.4f}")
    save_json(os.path.join(out_dir, "sim_metrics.json"), {
        "sim_runs": sim_runs,
        "ppo_mean": float(np.mean(ppo_costs)), "ppo_std": float(np.std(ppo_costs)),
        "hun_mean": float(np.mean(hun_costs)), "hun_std": float(np.std(hun_costs)),
        "gap_mean": float(dg.mean()),          "gap_std": float(dg.std()),
    })


# ─────────────────────────────────────────────────────────────────────────────
# AirSim evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_airsim(
    model_path, tests=30, out_dir="outputs/ppo",
    task_multiplier=1, weights=None
):
    try:
        import airsim
        from air_helpers import (
            get_airsim_scene, reached, prepare_drones, shutdown_drones,
        )
    except ImportError:
        print("AirSim not available – use --mode sim instead."); return

    os.makedirs(out_dir, exist_ok=True)
    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.model.eval()

    client      = airsim.MultirotorClient(); client.confirmConnection()
    drone_names = ["Drone1", "Drone2", "Drone3"]

    all_gaps, all_times, success_flags = [], [], []

    for idx in range(tests):
        print(f"\n===== AirSim TEST {idx+1}/{tests} =====")
        prepare_drones(client, drone_names)
        drones, drone_speeds = get_airsim_scene(client, drone_names)
        tasks = env.sample_tasks()
        drone_battery = np.ones(env.n, dtype=np.float32)
        task_priority = np.random.randint(1, 6, env.num_tasks).astype(np.float32)
        env.reset_from_scene(drones, tasks, drone_battery, drone_speeds, task_priority)

        state  = env.encode_state(); chosen = []
        for _ in range(env.n):
            valid   = env.get_valid_actions()
            action, _, _ = agent.select_action(state, valid, deterministic=True)
            chosen.append(action)
            state, _, _, _ = env.step(action)

        ppo_cost = sum(env.pair_cost(i, t) for i, t in enumerate(chosen))
        asgn, cm = assign_tasks(drones, tasks,
                                drone_battery=drone_battery,
                                task_priority=task_priority,
                                drone_speed=drone_speeds,
                                weights=env.weights)
        hun_cost = sum(cm[d][t] for d, t in asgn)
        all_gaps.append(ppo_cost - hun_cost)
        print(f"  PPO={ppo_cost:.4f}  Hungarian={hun_cost:.4f}  gap={ppo_cost-hun_cost:.4f}")

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
    parser.add_argument("--mode", choices=["train", "run", "sim"], required=True)
    parser.add_argument("--model_path",      default="ppo_assignment.pt")
    parser.add_argument("--episodes",        type=int, default=10_000)
    parser.add_argument("--tests",           type=int, default=30)
    parser.add_argument("--sim_runs",        type=int, default=1000)
    parser.add_argument("--out_dir",         default="outputs/ppo")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--task_multiplier", type=int, default=1)
    parser.add_argument("--weights",         type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    weights = parse_weights(args.weights)

    if args.mode == "train":
        train_ppo(args.model_path, args.episodes, args.out_dir,
                  args.task_multiplier, weights)
    elif args.mode == "sim":
        eval_sim(args.model_path, args.sim_runs, args.out_dir,
                 args.task_multiplier, weights)
    else:
        run_airsim(args.model_path, args.tests, args.out_dir,
                   args.task_multiplier, weights)