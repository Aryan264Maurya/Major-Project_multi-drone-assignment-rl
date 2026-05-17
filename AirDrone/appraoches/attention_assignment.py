"""
attention_assignment.py
=======================
Attention-based actor-critic for drone task assignment.

Motivation
----------
The MLP-based PPO and DQN agents use a flat state vector whose dimension
grows as O(n·m).  Attention-based policies are permutation-equivariant and
scale naturally: adding drones or tasks only changes the sequence length,
not the network width.

Architecture
------------
                ┌─────────────────────────────────────────┐
  drone feats   │  DroneEncoder (Linear → LayerNorm)      │
  [B, n, 8]  ──►│  → drone_tokens [B, n, d_model]         │
                └─────────────────────────────────────────┘
                ┌─────────────────────────────────────────┐
  task feats    │  TaskEncoder  (Linear → LayerNorm)      │
  [B, m, 6]  ──►│  → task_tokens [B, m, d_model]          │
                └─────────────────────────────────────────┘
                          concat → [B, n+m, d_model]
                ┌─────────────────────────────────────────┐
                │  Transformer Encoder (n_layers × n_head) │
                │  (self-attention over drones AND tasks)  │
                └──────────────┬──────────────────────────┘
                               │
              ┌────────────────┴──────────────────┐
              │                                   │
        task_tokens                         drone_tokens
        [B, m, d_model]                    [B, n, d_model]
              │                                   │
     actor_head (Linear→1)             global_avg_pool
        [B, m] logits                       value_head → [B]
              │
       masked softmax
              │
           action

Usage
-----
    python attention_assignment.py --mode train --episodes 10000
    python attention_assignment.py --mode sim   --model_path outputs/attn/models/attn.pt --sim_runs 1000
    python attention_assignment.py --mode run   --model_path outputs/attn/models/attn.pt --tests 30
"""

from __future__ import annotations
import os, argparse, random, time, json
from typing import Dict, List, Optional, Tuple

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

# Feature dimensions per agent / task (from AssignmentEnv.encode_state layout)
DRONE_FEAT_DIM = 8   # pos(3) + battery(1) + speed(3) + one_hot(1)
TASK_FEAT_DIM  = 6   # pos(3) + priority(1) + done(1) + dist(1)


# ─────────────────────────────────────────────────────────────────────────────
# State parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_state(flat_state: np.ndarray, n: int, m: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose the flat AssignmentEnv state into structured drone and task features.

    Layout (from AssignmentEnv.encode_state):
        drone_pos   : 3·n
        battery     : n
        speed       : 3·n
        task_pos    : 3·m
        priority    : m
        task_done   : m
        one_hot     : n
        dist_row    : m
    """
    s     = flat_state  # (state_dim,)
    idx   = 0

    drone_pos  = s[idx: idx + 3*n].reshape(n, 3);   idx += 3*n
    battery    = s[idx: idx + n  ].reshape(n, 1);   idx += n
    speed      = s[idx: idx + 3*n].reshape(n, 3);   idx += 3*n
    task_pos   = s[idx: idx + 3*m].reshape(m, 3);   idx += 3*m
    priority   = s[idx: idx + m  ].reshape(m, 1);   idx += m
    task_done  = s[idx: idx + m  ].reshape(m, 1);   idx += m
    one_hot    = s[idx: idx + n  ].reshape(n, 1);   idx += n
    dist_row   = s[idx: idx + m  ].reshape(m, 1);   idx += m

    drone_feats = np.concatenate([drone_pos, battery, speed, one_hot], axis=1)  # (n,8)
    task_feats  = np.concatenate([task_pos, priority, task_done, dist_row], axis=1)  # (m,6)

    return (
        torch.tensor(drone_feats, dtype=torch.float32),
        torch.tensor(task_feats,  dtype=torch.float32),
    )


def batch_parse_state(flat_states: np.ndarray, n: int, m: int):
    """Parse a batch of flat states → (B,n,8), (B,m,6)."""
    B = flat_states.shape[0]
    drone_list, task_list = [], []
    for i in range(B):
        d, t = parse_state(flat_states[i], n, m)
        drone_list.append(d); task_list.append(t)
    return torch.stack(drone_list), torch.stack(task_list)


# ─────────────────────────────────────────────────────────────────────────────
# Attention-based Actor-Critic
# ─────────────────────────────────────────────────────────────────────────────

class AttentionActorCritic(nn.Module):
    """
    Permutation-equivariant actor-critic using Transformer encoder.

    Scalability: n and m are not fixed in the architecture – only the
    embedding dimension d_model is fixed.  Adding more drones or tasks
    only changes the sequence length fed to the Transformer.
    """

    def __init__(
        self,
        drone_feat_dim: int = DRONE_FEAT_DIM,
        task_feat_dim:  int = TASK_FEAT_DIM,
        d_model:        int = 128,
        n_heads:        int = 4,
        n_layers:       int = 3,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Per-token encoders
        self.drone_encoder = nn.Sequential(
            nn.Linear(drone_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Learnable type embeddings (drone vs task)
        self.drone_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.task_type_emb  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.drone_type_emb, std=0.02)
        nn.init.normal_(self.task_type_emb,  std=0.02)

        # Transformer (self-attention over concatenated drone+task tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),   # per-task logit
        )
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),   # global scalar value
        )

    def forward(
        self,
        drone_feats: torch.Tensor,   # (B, n, drone_feat_dim)
        task_feats:  torch.Tensor,   # (B, m, task_feat_dim)
        task_mask:   Optional[torch.Tensor] = None,  # (B, m) bool, True = invalid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : (B, m)  – unnormalised log-probabilities per task
        value  : (B,)    – state value estimate
        """
        B, n, _ = drone_feats.shape
        m       = task_feats.shape[1]

        # Embed + add type tokens
        d_emb = self.drone_encoder(drone_feats) + self.drone_type_emb  # (B,n,d)
        t_emb = self.task_encoder(task_feats)   + self.task_type_emb   # (B,m,d)

        # Concatenate into one sequence
        tokens = torch.cat([d_emb, t_emb], dim=1)          # (B, n+m, d)
        tokens = self.transformer(tokens)                    # (B, n+m, d)

        drone_out = tokens[:, :n, :]                         # (B, n, d)
        task_out  = tokens[:, n:, :]                         # (B, m, d)

        # Actor: per-task logit
        logits = self.actor_head(task_out).squeeze(-1)       # (B, m)
        if task_mask is not None:
            logits = logits.masked_fill(task_mask, -1e9)

        # Critic: mean of drone tokens → global value
        value = self.critic_head(drone_out.mean(dim=1)).squeeze(-1)  # (B,)

        return logits, value


# ─────────────────────────────────────────────────────────────────────────────
# Agent wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PPOBuffer:
    def __init__(self): self.clear()
    def clear(self):
        self.drone_feats = []; self.task_feats = []
        self.actions = []; self.logprobs = []; self.rewards = []
        self.values  = []; self.dones    = []
    def add(self, df, tf, a, lp, r, v, d):
        self.drone_feats.append(df); self.task_feats.append(tf)
        self.actions.append(a);  self.logprobs.append(lp)
        self.rewards.append(r);  self.values.append(v); self.dones.append(d)
    def __len__(self): return len(self.actions)


class AttentionAgent:
    def __init__(
        self, n: int, m: int,
        d_model=128, n_heads=4, n_layers=3,
        lr=3e-4, gamma=0.99,
        clip_range=0.2, entropy_coef=0.01,
        value_coef=0.5, update_epochs=10,
        minibatch_size=32,
    ):
        self.n = n; self.m = m
        self.gamma        = gamma
        self.clip_range   = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.update_epochs  = update_epochs
        self.minibatch_size = minibatch_size

        self.model = AttentionActorCritic(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = PPOBuffer()

    def _flat_to_tensors(self, state: np.ndarray):
        df, tf = parse_state(state, self.n, self.m)
        return df.unsqueeze(0).to(DEVICE), tf.unsqueeze(0).to(DEVICE)

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Optional[List[int]] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        df, tf = self._flat_to_tensors(state)
        with torch.no_grad():
            # Build task mask (True = invalid/done)
            mask = None
            if valid_actions is not None:
                mask = torch.ones(1, self.m, dtype=torch.bool, device=DEVICE)
                mask[0, valid_actions] = False

            logits, value = self.model(df, tf, mask)
            dist   = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, 1) if deterministic else dist.sample()
            lp     = dist.log_prob(action)

        return int(action.item()), float(lp.item()), float(value.item())

    def update(self) -> Optional[Dict]:
        if len(self.buffer) == 0:
            return None

        B = len(self.buffer)
        drone_feats = torch.tensor(np.array(self.buffer.drone_feats),
                                   dtype=torch.float32, device=DEVICE)  # (B,n,8)
        task_feats  = torch.tensor(np.array(self.buffer.task_feats),
                                   dtype=torch.float32, device=DEVICE)  # (B,m,6)
        actions    = torch.tensor(self.buffer.actions,  dtype=torch.long,    device=DEVICE)
        old_lp     = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=DEVICE)
        rewards    = torch.tensor(self.buffer.rewards,  dtype=torch.float32, device=DEVICE)
        values     = torch.tensor(self.buffer.values,   dtype=torch.float32, device=DEVICE)
        dones      = torch.tensor(self.buffer.dones,    dtype=torch.float32, device=DEVICE)

        returns = torch.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(B)):
            running    = rewards[t] + self.gamma * running * (1.0 - dones[t])
            returns[t] = running

        advantages = returns - values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totals = dict(actor_loss=0.0, value_loss=0.0, entropy=0.0)
        n_upd  = 0
        idxs   = np.arange(B)
        bs     = min(self.minibatch_size, B)

        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, B, bs):
                mb  = idxs[start:start+bs]
                lg, vp = self.model(drone_feats[mb], task_feats[mb])
                dist   = torch.distributions.Categorical(logits=lg)
                new_lp = dist.log_prob(actions[mb])
                ent    = dist.entropy().mean()

                ratio      = torch.exp(new_lp - old_lp[mb])
                adv        = advantages[mb]
                actor_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * adv
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
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def rolling_mean(values, window=50):
    out = []
    for i in range(len(values)):
        s = max(0, i-window+1)
        out.append(float(np.mean(values[s:i+1])))
    return out


def print_stats(label, costs, times=None):
    costs = np.array(costs); n = len(costs)
    se    = costs.std() / np.sqrt(n)
    t_s, p_v = stats.ttest_1samp(costs, 0.0)
    print(f"\n===== {label} SUMMARY =====")
    print(f"Runs              : {n}")
    print(f"Cost  mean ± std  : {costs.mean():.4f} ± {costs.std():.4f}")
    print(f"Cost  95 % CI     : [{costs.mean()-1.96*se:.4f}, {costs.mean()+1.96*se:.4f}]")
    print(f"Cost  [min, max]  : [{costs.min():.4f}, {costs.max():.4f}]")
    if times:
        t = np.array(times)
        print(f"Time  mean ± std  : {t.mean():.2f} ± {t.std():.2f} s")
    print(f"t-test vs 0       : t={t_s:.3f}, p={p_v:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(history, out_dir, total_episodes):
    os.makedirs(out_dir, exist_ok=True)
    ep = np.arange(1, total_episodes + 1)
    assert len(history["reward"]) == total_episodes

    for key, ylabel, title, fname in [
        ("reward", "Avg reward",   "Attention Reward Convergence",         "reward_convergence.png"),
        ("cost",   "Avg cost",     "Attention Cost Convergence",           "cost_convergence.png"),
        ("gap",    "Avg cost gap", "Attention Optimality Gap Convergence", "gap_convergence.png"),
    ]:
        plt.figure(figsize=(10,5))
        plt.plot(ep, rolling_mean(history[key], 50))
        plt.xlabel("Episode"); plt.ylabel(ylabel); plt.title(title)
        plt.xlim(1, total_episodes); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200); plt.close()

    if history["loss"]:
        up = np.arange(1, len(history["loss"])+1)
        plt.figure(figsize=(10,5))
        plt.plot(up, rolling_mean(history["loss"], 20))
        plt.xlabel("PPO update"); plt.ylabel("Loss"); plt.title("Attention PPO Loss")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "attn_loss.png"), dpi=200); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_attention(
    model_path="attn_assignment.pt",
    episodes=10_000,
    out_dir="outputs/attn",
    task_multiplier=1,
    weights=None,
    d_model=128,
    n_heads=4,
    n_layers=3,
):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir  = os.path.join(out_dir, "plots");  os.makedirs(plots_dir,  exist_ok=True)
    models_dir = os.path.join(out_dir, "models"); os.makedirs(models_dir, exist_ok=True)

    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    n, m  = env.n, env.num_tasks
    agent = AttentionAgent(n, m, d_model=d_model, n_heads=n_heads, n_layers=n_layers)

    config = {
        "algo":              "attention_ppo",
        "episodes":          episodes,
        "task_multiplier":   task_multiplier,
        "n": n, "m": m,
        "d_model":           d_model,
        "n_heads":           n_heads,
        "n_layers":          n_layers,
        "weights":           env.weights,
        "problem_signature": env.problem_signature,
        "problem_hash":      env.problem_hash,
    }
    save_json(os.path.join(out_dir, "config.json"), config)
    print(f"Problem hash: {env.problem_hash}")

    history = {"reward": [], "cost": [], "gap": [], "loss": []}

    for ep in range(episodes):
        state = env.reset()
        done  = False; ep_reward = 0.0

        while not done:
            valid   = env.get_valid_actions()
            action, lp, v = agent.select_action(state, valid, deterministic=False)

            # Store structured features for the attention buffer
            df, tf = parse_state(state, n, m)
            ns, r, done, _ = env.step(action)
            agent.buffer.add(
                df.numpy(), tf.numpy(), action, lp, r, v, done
            )
            state = ns; ep_reward += r

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

        if (ep + 1) % 2500 == 0:
            ck = os.path.join(models_dir, f"attn_ep{ep+1}.pt")
            torch.save(agent.model.state_dict(), ck)
            print(f"  Checkpoint → {ck}")

    save_path = os.path.join(models_dir, model_path)
    torch.save(agent.model.state_dict(), save_path)
    print(f"Model saved → {save_path}")
    plot_convergence(history, plots_dir, episodes)


# ─────────────────────────────────────────────────────────────────────────────
# Simulator evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_sim(
    model_path, sim_runs=1000, out_dir="outputs/attn",
    task_multiplier=1, weights=None,
    d_model=128, n_heads=4, n_layers=3,
):
    os.makedirs(out_dir, exist_ok=True)
    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    n, m  = env.n, env.num_tasks
    agent = AttentionAgent(n, m, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.model.eval()

    print(f"Problem hash: {env.problem_hash}")
    attn_costs, hun_costs, gaps = [], [], []

    for _ in range(sim_runs):
        state  = env.reset(); chosen = []
        for _ in range(env.n):
            valid   = env.get_valid_actions()
            action, _, _ = agent.select_action(state, valid, deterministic=True)
            chosen.append(action)
            state, _, _, _ = env.step(action)

        attn_cost = sum(env.pair_cost(i, t) for i, t in enumerate(chosen))
        asgn, cm  = assign_tasks(env.drones, env.tasks,
                                  drone_battery=env.drone_battery,
                                  task_priority=env.task_priority,
                                  drone_speed=env.drone_speed,
                                  weights=env.weights)
        hun_cost  = sum(cm[d][t] for d, t in asgn)
        attn_costs.append(attn_cost); hun_costs.append(hun_cost)
        gaps.append(attn_cost - hun_cost)

    print_stats(f"ATTENTION SIM ({sim_runs} runs)", attn_costs)
    print_stats("HUNGARIAN SIM",                    hun_costs)
    dg = np.array(gaps)
    print(f"\nAttention − Hungarian  mean={dg.mean():.4f} ± {dg.std():.4f}")
    save_json(os.path.join(out_dir, "sim_metrics.json"), {
        "sim_runs":  sim_runs,
        "attn_mean": float(np.mean(attn_costs)), "attn_std": float(np.std(attn_costs)),
        "hun_mean":  float(np.mean(hun_costs)),  "hun_std":  float(np.std(hun_costs)),
        "gap_mean":  float(dg.mean()),            "gap_std":  float(dg.std()),
    })


# ─────────────────────────────────────────────────────────────────────────────
# AirSim evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_airsim(
    model_path, tests=30, out_dir="outputs/attn",
    task_multiplier=1, weights=None,
    d_model=128, n_heads=4, n_layers=3,
):
    try:
        import airsim
        from air_helpers import (
            get_airsim_scene, reached, prepare_drones, shutdown_drones,
        )
    except ImportError:
        print("AirSim not available – use --mode sim."); return

    os.makedirs(out_dir, exist_ok=True)
    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    n, m  = env.n, env.num_tasks
    agent = AttentionAgent(n, m, d_model=d_model, n_heads=n_heads, n_layers=n_layers)
    agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.model.eval()

    client = airsim.MultirotorClient(); client.confirmConnection()
    drone_names = ["Drone1", "Drone2", "Drone3"]

    all_gaps, all_times, success_flags = [], [], []

    for idx in range(tests):
        print(f"\n===== Attention TEST {idx+1}/{tests} =====")
        prepare_drones(client, drone_names)
        drones, drone_speeds = get_airsim_scene(client, drone_names)
        tasks = env.sample_tasks()
        drone_battery = np.ones(env.n, dtype=np.float32)
        task_priority = np.random.randint(1, 6, env.num_tasks).astype(np.float32)
        env.reset_from_scene(drones, tasks, drone_battery, drone_speeds, task_priority)

        state = env.encode_state(); chosen = []
        for _ in range(env.n):
            valid   = env.get_valid_actions()
            action, _, _ = agent.select_action(state, valid, deterministic=True)
            chosen.append(action)
            state, _, _, _ = env.step(action)

        attn_cost = sum(env.pair_cost(i, t) for i, t in enumerate(chosen))
        asgn, cm  = assign_tasks(drones, tasks,
                                  drone_battery=drone_battery,
                                  task_priority=task_priority,
                                  drone_speed=drone_speeds,
                                  weights=env.weights)
        hun_cost = sum(cm[d][t] for d, t in asgn)
        all_gaps.append(attn_cost - hun_cost)
        print(f"  Attention={attn_cost:.4f}  Hungarian={hun_cost:.4f}  "
              f"gap={attn_cost-hun_cost:.4f}")

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
    parser.add_argument("--mode", choices=["train","sim","run"], required=True)
    parser.add_argument("--model_path",      default="attn_assignment.pt")
    parser.add_argument("--episodes",        type=int, default=10_000)
    parser.add_argument("--tests",           type=int, default=30)
    parser.add_argument("--sim_runs",        type=int, default=1000)
    parser.add_argument("--out_dir",         default="outputs/attn")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--task_multiplier", type=int, default=1)
    parser.add_argument("--d_model",         type=int, default=128)
    parser.add_argument("--n_heads",         type=int, default=4)
    parser.add_argument("--n_layers",        type=int, default=3)
    parser.add_argument("--weights",         type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    weights = None if args.weights is None else json.loads(args.weights)

    if args.mode == "train":
        train_attention(args.model_path, args.episodes, args.out_dir,
                        args.task_multiplier, weights,
                        args.d_model, args.n_heads, args.n_layers)
    elif args.mode == "sim":
        eval_sim(args.model_path, args.sim_runs, args.out_dir,
                 args.task_multiplier, weights,
                 args.d_model, args.n_heads, args.n_layers)
    else:
        run_airsim(args.model_path, args.tests, args.out_dir,
                   args.task_multiplier, weights,
                   args.d_model, args.n_heads, args.n_layers)