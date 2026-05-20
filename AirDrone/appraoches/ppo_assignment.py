"""
ppo_assignment.py  (fully fixed)
=================================
All fixes applied:

  FIX 1 – Rollout accumulation:
      End-of-episode flush fired every episode with only 3 transitions,
      making effective rollout = 3 instead of 256.  Buffer now accumulates
      ACROSS episodes and only updates when >= update_every steps are stored.
      The end-of-episode flush is REMOVED from the inner loop.

  FIX 2 – Mask in update phase:
      During the PPO gradient update, log_prob was recomputed WITHOUT the
      action validity mask, giving incorrect probability ratios and biased
      gradients.  The valid-action mask is now stored in the buffer for
      every step and reapplied when computing new_logprobs during updates.

  FIX 3 – GAE instead of Monte-Carlo returns:
      Monte-Carlo returns have very high variance for a flat MLP critic,
      drowning the gradient signal.  Replaced with Generalised Advantage
      Estimation (lambda=0.95), which trades a small amount of bias for a
      large variance reduction and produces cleaner advantage estimates.

  FIX 4 – Running observation normalisation:
      The flat state vector mixes features at very different scales
      (positions O(100), battery O(1), priority O(1-5)).  A lightweight
      online RunningNorm (mean/std) clips to [-10, 10] and is applied to
      every observation before it enters the network.

  FIX 5 – Entropy coefficient decay:
      A fixed entropy_coef=0.05 is too high in late training, preventing
      the policy from committing to good actions.  The coefficient now
      decays linearly from 0.05 → 0.005 over the training run so the
      agent explores early and exploits late.

Hyperparameters (otherwise unchanged):
  clip_range=0.25, update_every=256, lr=1e-4, mb=64, gae_lambda=0.95
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


# ── Utilities ─────────────────────────────────────────────────────────────────

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


# ── FIX 4: Running observation normaliser ────────────────────────────────────

class RunningNorm:
    """
    Online mean/std normaliser for flat observation vectors.
    Updates incrementally; clips output to [-clip, clip].
    """
    def __init__(self, dim: int, clip: float = 10.0):
        self.mean  = np.zeros(dim, dtype=np.float64)
        self.var   = np.ones(dim,  dtype=np.float64)
        self.count = 1e-4
        self.clip  = clip

    def update(self, x: np.ndarray):
        self.count += 1
        delta       = x - self.mean
        self.mean  += delta / self.count
        self.var   += (delta * (x - self.mean) - self.var) / self.count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        normed = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)


# ── PPO Buffer (stores mask per step) ────────────────────────────────────────

class PPOBuffer:
    """
    Stores transitions across MULTIPLE episodes until update_every is reached.
    Each transition stores the boolean valid-action mask so it can be
    reapplied during the PPO update phase (Fix 2).
    """
    def __init__(self): self.clear()

    def clear(self):
        self.states   = []
        self.actions  = []
        self.logprobs = []
        self.rewards  = []
        self.values   = []
        self.dones    = []
        self.masks    = []   # FIX 2: per-step valid-action mask

    def add(self, s, a, lp, r, v, d, mask):
        """mask: boolean tensor of shape (action_dim,), True = VALID action."""
        self.states.append(s)
        self.actions.append(a)
        self.logprobs.append(lp)
        self.rewards.append(r)
        self.values.append(v)
        self.dones.append(d)
        self.masks.append(mask)   # FIX 2

    def __len__(self): return len(self.states)


# ── Actor-Critic network ──────────────────────────────────────────────────────

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


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(self, state_dim, action_dim,
                 lr=1e-4,
                 gamma=0.99,
                 gae_lambda=0.95,       # FIX 3
                 clip_range=0.25,
                 entropy_coef=0.05,
                 value_coef=0.5,
                 update_epochs=10,
                 minibatch_size=64):
        self.action_dim     = action_dim
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda     # FIX 3
        self.clip_range     = clip_range
        self.entropy_coef   = entropy_coef   # FIX 5: mutated externally during training
        self.value_coef     = value_coef
        self.update_epochs  = update_epochs
        self.minibatch_size = minibatch_size

        self.model     = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer    = PPOBuffer()
        self.obs_norm  = RunningNorm(state_dim)   # FIX 4

    def _preprocess(self, state: np.ndarray) -> np.ndarray:
        """FIX 4: update running stats and return normalised state."""
        self.obs_norm.update(state)
        return self.obs_norm.normalize(state)

    def select_action(self, state, valid_actions=None, deterministic=False):
        """
        Returns (action, log_prob, value, valid_mask).
        Normalises the observation before inference (FIX 4).
        valid_mask is stored in the buffer for FIX 2.
        """
        norm_state = self._preprocess(state)   # FIX 4
        st = torch.tensor(norm_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Build boolean mask: True = valid
        valid_mask = torch.zeros(self.action_dim, dtype=torch.bool)
        if valid_actions is not None:
            valid_mask[valid_actions] = True
        else:
            valid_mask[:] = True

        with torch.no_grad():
            logits, value = self.model(st)
            masked_logits = logits.clone()
            masked_logits[0, ~valid_mask.to(DEVICE)] = -1e9
            dist   = torch.distributions.Categorical(logits=masked_logits)
            action = torch.argmax(masked_logits, 1) if deterministic else dist.sample()
            lp     = dist.log_prob(action)

        return int(action.item()), float(lp.item()), float(value.item()), valid_mask

    def update(self):
        if len(self.buffer) == 0:
            return None

        B = len(self.buffer)

        # FIX 4: states were already normalised when stored
        states   = torch.tensor(np.array(self.buffer.states),   dtype=torch.float32, device=DEVICE)
        actions  = torch.tensor(np.array(self.buffer.actions),  dtype=torch.long,    device=DEVICE)
        old_lp   = torch.tensor(np.array(self.buffer.logprobs), dtype=torch.float32, device=DEVICE)
        rewards  = torch.tensor(np.array(self.buffer.rewards),  dtype=torch.float32, device=DEVICE)
        values   = torch.tensor(np.array(self.buffer.values),   dtype=torch.float32, device=DEVICE)
        dones    = torch.tensor(np.array(self.buffer.dones),    dtype=torch.float32, device=DEVICE)
        # FIX 2: stack stored masks → (B, action_dim)
        masks    = torch.stack(self.buffer.masks).to(DEVICE)

        # ── FIX 3: GAE instead of Monte-Carlo returns ──────────────────────
        advantages = torch.zeros_like(rewards)
        last_gae   = 0.0
        for t in reversed(range(B)):
            next_val   = float(values[t + 1].item()) if t + 1 < B else 0.0
            delta      = (rewards[t]
                          + self.gamma * next_val * (1.0 - dones[t])
                          - values[t])
            last_gae   = (float(delta.item())
                          + self.gamma * self.gae_lambda
                          * (1.0 - float(dones[t].item())) * last_gae)
            advantages[t] = last_gae
        returns = advantages + values   # value loss target
        # ───────────────────────────────────────────────────────────────────

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        totals = dict(actor_loss=0.0, value_loss=0.0, entropy=0.0)
        n_upd  = 0
        idxs   = np.arange(B)
        bs     = min(self.minibatch_size, B)

        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, B, bs):
                mb = idxs[start:start + bs]

                logits, vp = self.model(states[mb])

                # FIX 2: reapply per-step mask before computing new log_probs
                mb_masks = masks[mb]
                invalid  = ~mb_masks
                logits   = logits.masked_fill(invalid, -1e9)

                dist    = torch.distributions.Categorical(logits=logits)
                new_lp  = dist.log_prob(actions[mb])
                ent     = dist.entropy().mean()

                ratio      = torch.exp(new_lp - old_lp[mb])
                adv        = advantages[mb]
                actor_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv
                ).mean()
                value_loss = F.mse_loss(vp, returns[mb])
                loss = (actor_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * ent)   # FIX 5: entropy_coef decayed externally

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                totals["actor_loss"] += float(actor_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"]    += float(ent.item())
                n_upd += 1

        self.buffer.clear()
        return {k: v / max(1, n_upd) for k, v in totals.items()}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_convergence(history, out_dir, total_episodes):
    os.makedirs(out_dir, exist_ok=True)
    assert len(history["reward"]) == total_episodes
    ep = np.arange(1, total_episodes + 1)
    for key, ylabel, title, fname in [
        ("reward", "Avg reward",  "PPO Reward Convergence",         "reward_convergence.png"),
        ("cost",   "Avg cost",    "PPO Cost Convergence",           "cost_convergence.png"),
        ("gap",    "Avg gap",     "PPO Optimality Gap Convergence", "gap_convergence.png"),
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
        plt.xlabel("PPO update"); plt.ylabel("Loss"); plt.title("PPO Training Loss")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ppo_loss.png"), dpi=200); plt.close()


# ── Training ──────────────────────────────────────────────────────────────────

def train_ppo(model_path="ppo_assignment.pt", episodes=10_000,
              update_every=256, out_dir="outputs/ppo",
              task_multiplier=1, weights=None):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir  = os.path.join(out_dir, "plots");  os.makedirs(plots_dir,  exist_ok=True)
    models_dir = os.path.join(out_dir, "models"); os.makedirs(models_dir, exist_ok=True)

    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    # FIX 5: entropy decay schedule
    entropy_start = 0.05
    entropy_end   = 0.005

    save_json(os.path.join(out_dir, "config.json"), {
        "algo": "ppo", "episodes": episodes, "update_every": update_every,
        "entropy_start": entropy_start, "entropy_end": entropy_end,
        "clip_range": 0.25, "lr": 1e-4, "minibatch_size": 64,
        "gae_lambda": 0.95,
        "weights": env.weights, "problem_hash": env.problem_hash,
        "problem_signature": env.problem_signature,
        "fix1_rollout_cross_episode": True,
        "fix2_mask_in_update": True,
        "fix3_gae": True,
        "fix4_obs_norm": True,
        "fix5_entropy_decay": True,
    })
    print(f"Problem hash  : {env.problem_hash}")
    print(f"PPO config    : clip=0.25  rollout={update_every}  lr=1e-4  mb=64  gae_lambda=0.95")
    print(f"FIX 1 active  : buffer accumulates ACROSS episodes (no per-episode flush)")
    print(f"FIX 2 active  : action mask stored & reapplied during PPO update")
    print(f"FIX 3 active  : GAE (lambda=0.95) replaces Monte-Carlo returns")
    print(f"FIX 4 active  : running obs normalisation (clip=10)")
    print(f"FIX 5 active  : entropy_coef decays {entropy_start} → {entropy_end} over training")

    history = {"reward": [], "cost": [], "gap": [], "loss": []}

    for ep in range(episodes):
        # FIX 5: update entropy coefficient for this episode
        frac = ep / max(1, episodes - 1)
        agent.entropy_coef = entropy_start - frac * (entropy_start - entropy_end)

        state     = env.reset()
        done      = False
        ep_reward = 0.0

        while not done:
            valid = env.get_valid_actions()
            # select_action normalises state internally (FIX 4)
            # and returns the valid_mask for buffer storage (FIX 2)
            action, lp, v, valid_mask = agent.select_action(
                state, valid_actions=valid, deterministic=False
            )
            # Store the normalised state (already done inside select_action via _preprocess)
            norm_state = agent.obs_norm.normalize(state)   # retrieve normalised version
            ns, r, done, _ = env.step(action)

            # FIX 1 + FIX 2: store normalised state and valid_mask
            agent.buffer.add(norm_state, action, lp, r, v, done, valid_mask)
            state = ns
            ep_reward += r

        # FIX 1: update only when buffer has accumulated update_every steps
        # (cross-episode accumulation — no per-episode flush)
        if len(agent.buffer) >= update_every:
            metrics = agent.update()
            if metrics is not None:
                history["loss"].append(
                    metrics["actor_loss"] + metrics["value_loss"] - metrics["entropy"]
                )

        history["reward"].append(ep_reward)
        history["cost"].append(-ep_reward)
        history["gap"].append(-ep_reward - env.optimal_cost())

        if (ep + 1) % 500 == 0:
            buf_size = len(agent.buffer)
            print(f"Ep {ep+1:5d} | "
                  f"avg_r={np.mean(history['reward'][-500:]):.3f} | "
                  f"avg_cost={np.mean(history['cost'][-500:]):.3f} | "
                  f"gap={np.mean(history['gap'][-500:]):.3f} | "
                  f"buf={buf_size}/{update_every} | "
                  f"ent={agent.entropy_coef:.4f}")

        if (ep + 1) % 2500 == 0:
            ck = os.path.join(models_dir, f"ppo_ep{ep+1}.pt")
            torch.save(agent.model.state_dict(), ck)
            print(f"  Checkpoint → {ck}")

    # Final flush of any remaining transitions at end of training
    if len(agent.buffer) > 0:
        metrics = agent.update()
        if metrics is not None:
            history["loss"].append(
                metrics["actor_loss"] + metrics["value_loss"] - metrics["entropy"]
            )

    save_path = os.path.join(models_dir, model_path)
    torch.save(agent.model.state_dict(), save_path)
    print(f"Model saved → {save_path}")
    plot_convergence(history, plots_dir, episodes)
    print(f"Plots saved → {plots_dir}")


# ── Simulator evaluation ──────────────────────────────────────────────────────

def eval_sim(model_path, sim_runs=1000, out_dir="outputs/ppo",
             task_multiplier=1, weights=None):
    os.makedirs(out_dir, exist_ok=True)
    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.model.eval()
    print(f"Problem hash: {env.problem_hash}")

    ppo_costs, hun_costs, gaps = [], [], []
    for _ in range(sim_runs):
        state  = env.reset(); chosen = []
        for _ in range(env.n):
            valid = env.get_valid_actions()
            action, _, _, _ = agent.select_action(state, valid, deterministic=True)
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
        ppo_costs.append(ppo_cost); hun_costs.append(hun_cost)
        gaps.append(ppo_cost - hun_cost)

    print_stats(f"PPO SIM ({sim_runs} runs)", ppo_costs)
    print_stats("HUNGARIAN SIM", hun_costs)
    dg = np.array(gaps)
    print(f"\nPPO − Hungarian  mean={dg.mean():.4f} ± {dg.std():.4f}")
    save_json(os.path.join(out_dir, "sim_metrics.json"), {
        "sim_runs": sim_runs,
        "ppo_mean": float(np.mean(ppo_costs)), "ppo_std": float(np.std(ppo_costs)),
        "hun_mean": float(np.mean(hun_costs)), "hun_std": float(np.std(hun_costs)),
        "gap_mean": float(dg.mean()),          "gap_std": float(dg.std()),
    })


# ── AirSim evaluation ─────────────────────────────────────────────────────────

def run_airsim(model_path, tests=30, out_dir="outputs/ppo",
               task_multiplier=1, weights=None):
    try:
        import airsim
        from air_helpers import get_airsim_scene, reached, prepare_drones, shutdown_drones
    except ImportError:
        print("AirSim not available – use --mode sim."); return

    os.makedirs(out_dir, exist_ok=True)
    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.model.eval()

    client = airsim.MultirotorClient(); client.confirmConnection()
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

        state = env.encode_state(); chosen = []
        for _ in range(env.n):
            valid = env.get_valid_actions()
            action, _, _, _ = agent.select_action(state, valid, deterministic=True)
            chosen.append(action)
            state, _, _, _ = env.step(action)

        ppo_cost = sum(env.pair_cost(i, t) for i, t in enumerate(chosen))
        asgn, cm = assign_tasks(
            drones, tasks, drone_battery=drone_battery,
            task_priority=task_priority, drone_speed=drone_speeds,
            weights=env.weights,
        )
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "run", "sim"], required=True)
    parser.add_argument("--model_path",      default="ppo_assignment.pt")
    parser.add_argument("--episodes",        type=int, default=10_000)
    parser.add_argument("--update_every",    type=int, default=256)
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
        train_ppo(args.model_path, args.episodes, args.update_every,
                  args.out_dir, args.task_multiplier, weights)
    elif args.mode == "sim":
        eval_sim(args.model_path, args.sim_runs, args.out_dir,
                 args.task_multiplier, weights)
    else:
        run_airsim(args.model_path, args.tests, args.out_dir,
                   args.task_multiplier, weights)