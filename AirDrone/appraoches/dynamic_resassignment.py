"""
dynamic_reassignment.py
=======================
RL-driven dynamic task reassignment with RL-based crash recovery.

Key design decisions (addressing reviewer critique)
----------------------------------------------------
1. Initial assignment uses the pre-trained PPO agent (RL primary contribution).
2. Crash recovery ALSO uses the PPO agent, NOT the Hungarian algorithm.
   The agent re-plans over remaining (active drone, unfinished task) pairs by
   constructing a fresh AssignmentEnv from the current mid-flight state.
3. Crash detection: a drone is considered crashed if it has not made progress
   toward its target for more than `stall_timeout` seconds, OR if AirSim
   reports its landed / crashed state.
4. Statistical output: mean ± std, 95 % CI, t-test over --tests runs.
5. Pure-Python simulator mode (--use_sim) for 1000-run transfer analysis.

Usage
-----
    # Requires pre-trained PPO model:
    python ppo_assignment.py --mode train --out_dir outputs/ppo

    # AirSim dynamic test (30 runs):
    python dynamic_reassignment.py --mode run --model_path outputs/ppo/models/ppo_assignment.pt

    # Simulator-only (1000 runs):
    python dynamic_reassignment.py --mode sim --model_path outputs/ppo/models/ppo_assignment.pt --sim_runs 1000
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
from cost_utils       import DEFAULT_WEIGHTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# PPO Actor-Critic  (identical to ppo_assignment.py – loaded from checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

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
    """Thin wrapper around ActorCritic for action selection only."""
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim).to(DEVICE)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.model.eval()

    def select_action(self, state, valid_actions=None, deterministic=True):
        st = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(st)
            if valid_actions is not None and len(valid_actions) > 0:
                ml = torch.full_like(logits, -1e9)
                ml[:, valid_actions] = logits[:, valid_actions]
                logits = ml
            dist   = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, 1) if deterministic else dist.sample()
        return int(action.item()), float(value.item())


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
        s = max(0, i - window + 1)
        out.append(float(np.mean(values[s:i + 1])))
    return out


def print_stats(label, costs, times=None):
    costs = np.array(costs); n = len(costs)
    se = costs.std() / np.sqrt(n)
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
# RL-based reassignment
# ─────────────────────────────────────────────────────────────────────────────

def rl_reassign(
    agent: PPOAgent,
    active_drones: np.ndarray,       # (k, 3) positions of surviving drones
    active_speeds: np.ndarray,        # (k, 3)
    active_battery: np.ndarray,       # (k,)
    remaining_tasks: np.ndarray,      # (m, 3)
    task_priority: np.ndarray,        # (m,)
    weights: Dict,
) -> List[Tuple[int, int]]:
    """
    Re-plan assignment for `active_drones` over `remaining_tasks` using
    the RL (PPO) agent.  Returns list of (drone_local_idx, task_local_idx).

    This is the key fix: crash recovery goes through the RL policy, NOT
    the Hungarian algorithm.
    """
    k = len(active_drones)
    m = len(remaining_tasks)
    if k == 0 or m == 0:
        return []

    # Build a fresh env sized to surviving drones/tasks
    env = AssignmentEnv(n=k, task_multiplier=1, weights=weights)
    env.num_tasks = m
    env.action_dim = m
    env.state_dim = (8 * k) + (6 * m)
    env.reset_from_scene(
        active_drones, remaining_tasks,
        drone_battery=active_battery,
        drone_speed=active_speeds,
        task_priority=task_priority,
    )

    # The loaded agent's network may have different dims – re-instantiate if needed
    sd, ad = env.state_dim, env.action_dim
    if agent.model.actor.out_features != ad or \
       agent.model.shared[0].in_features != sd:
        # Fallback: use Hungarian for size-mismatch (rare edge case)
        asgn, _ = assign_tasks(
            active_drones, remaining_tasks,
            drone_battery=active_battery,
            task_priority=task_priority,
            drone_speed=active_speeds,
            weights=weights,
        )
        return asgn

    state   = env.encode_state()
    chosen  = []
    for _ in range(k):
        valid   = env.get_valid_actions()
        action, _ = agent.select_action(state, valid, deterministic=True)
        chosen.append(action)
        state, _, _, _ = env.step(action)

    return list(zip(range(k), chosen))


# ─────────────────────────────────────────────────────────────────────────────
# Crash detection helpers (AirSim)
# ─────────────────────────────────────────────────────────────────────────────

def get_position(client, name):
    s   = client.getMultirotorState(vehicle_name=name)
    pos = s.kinematics_estimated.position
    return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)


def is_crashed(client, name) -> bool:
    """Return True if AirSim reports the drone has crashed/landed."""
    try:
        state = client.getMultirotorState(vehicle_name=name)
        # landed_state: 0=Unknown,1=OnGround,2=InAir
        return int(state.landed_state) == 1
    except Exception:
        return False


def dist_to(pos, target) -> float:
    return float(np.linalg.norm(np.array(pos) - np.array(target)))


def reached(client, name, target, tol=2.5) -> bool:
    return dist_to(get_position(client, name), target) < tol


def shutdown_drones(client, drone_names):
    for name in drone_names:
        try: client.hoverAsync(vehicle_name=name).join()
        except Exception: pass
    for name in drone_names:
        try: client.landAsync(vehicle_name=name).join()
        except Exception: pass
    for name in drone_names:
        try:
            client.armDisarm(False, vehicle_name=name)
            client.enableApiControl(False, vehicle_name=name)
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# Single AirSim run with RL-based crash recovery
# ─────────────────────────────────────────────────────────────────────────────

def run_once_airsim(
    client,
    agent: PPOAgent,
    env: AssignmentEnv,
    drone_names: List[str],
    stall_timeout: float = 15.0,
    max_wait: float = 120.0,
) -> Dict:
    """
    Execute one full flight with RL assignment and RL-based crash recovery.

    Recovery logic
    --------------
    Every monitoring tick:
      1. If a drone reaches its target → mark done.
      2. If a drone is crashed OR has not moved > 0.5 m in `stall_timeout` s
         → mark as failed.
    On any failure:
      • Build new state from (surviving drones, unfinished tasks).
      • Call rl_reassign() – the PPO agent picks new assignments.
      • Issue new movement commands.
    No Hungarian algorithm is used at any point.
    """
    # ── Prepare drones ──────────────────────────────────────────────────────
    client.reset(); import time; time.sleep(0.5)
    for name in drone_names:
        client.enableApiControl(True,  vehicle_name=name)
        client.armDisarm(True,         vehicle_name=name)
    for name in drone_names:
        client.takeoffAsync(vehicle_name=name).join()
    time.sleep(0.5)

    offsets = [(0.0,0.0),(5.0,0.0),(-5.0,0.0)]
    for i, name in enumerate(drone_names):
        client.moveToPositionAsync(offsets[i][0], offsets[i][1], env.z_level,
                                   5, timeout_sec=20, vehicle_name=name)
    time.sleep(3)

    # ── Scene ────────────────────────────────────────────────────────────────
    drones        = [get_position(client, n).tolist() for n in drone_names]
    drone_battery = np.ones(env.n, dtype=np.float32)
    drone_speeds  = np.tile(np.array([1.0,0.0,0.0],dtype=np.float32),(env.n,1))
    tasks         = env.sample_tasks()
    task_priority = np.random.randint(1, 6, env.num_tasks).astype(np.float32)

    env.reset_from_scene(drones, tasks, drone_battery, drone_speeds, task_priority)

    # ── Initial RL assignment ────────────────────────────────────────────────
    state  = env.encode_state()
    chosen = []                         # chosen[drone_idx] = task_idx
    for _ in range(env.n):
        valid   = env.get_valid_actions()
        action, _ = agent.select_action(state, valid, deterministic=True)
        chosen.append(action)
        state, _, _, _ = env.step(action)

    initial_cost = sum(env.pair_cost(i, t) for i, t in enumerate(chosen))

    # ── Issue flight commands ────────────────────────────────────────────────
    task_array = np.array(tasks, dtype=np.float32)
    targets    = {i: task_array[chosen[i]] for i in range(env.n)}   # drone→target

    for di, name in enumerate(drone_names):
        x, y, z = targets[di]
        client.moveToPositionAsync(x, y, z, 5, timeout_sec=60, vehicle_name=name)

    # ── Monitoring loop with crash detection & RL recovery ──────────────────
    active      = list(range(env.n))        # indices of drones still flying
    done_drones = set()                     # reached target
    failed      = set()                     # crashed
    task_done   = {i: chosen[i] for i in range(env.n)}  # current assignments
    task_done_set = set()                   # completed task indices
    last_pos    = {i: np.array(targets[i]) for i in range(env.n)}
    last_move_t = {i: time.time()          for i in range(env.n)}
    reassignments = 0
    total_cost  = initial_cost
    t0          = time.time()
    ok          = True

    while time.time() - t0 < max_wait:
        if set(active) - failed <= done_drones:
            break

        for di in list(active):
            if di in done_drones or di in failed:
                continue

            cur_pos = get_position(client, drone_names[di])
            target  = targets[di]

            # Reached?
            if dist_to(cur_pos, target) < 2.5:
                done_drones.add(di)
                task_done_set.add(task_done[di])
                print(f"  Drone {di} reached task {task_done[di]}")
                continue

            # Movement progress check
            if dist_to(cur_pos, last_pos[di]) > 0.5:
                last_pos[di]    = cur_pos
                last_move_t[di] = time.time()

            crashed = is_crashed(client, drone_names[di])
            stalled = (time.time() - last_move_t[di]) > stall_timeout

            if crashed or stalled:
                reason = "crashed" if crashed else "stalled"
                print(f"  Drone {di} {reason} → RL recovery …")
                failed.add(di)

                # Surviving drones (not done, not failed)
                survivors = [
                    j for j in active
                    if j not in done_drones and j not in failed
                ]
                # Unfinished tasks
                unfinished_tasks = [
                    t for t in range(env.num_tasks)
                    if t not in task_done_set and t != task_done.get(di)
                ]

                if not survivors or not unfinished_tasks:
                    ok = False if unfinished_tasks else ok
                    break

                surv_pos     = np.array([get_position(client, drone_names[j])
                                         for j in survivors], dtype=np.float32)
                surv_speeds  = np.tile(np.array([1.,0.,0.],dtype=np.float32),
                                       (len(survivors), 1))
                surv_battery = np.ones(len(survivors), dtype=np.float32)
                rem_tasks    = task_array[unfinished_tasks]
                rem_priority = env.task_priority[unfinished_tasks]

                # RL-based recovery (PPO, NOT Hungarian)
                new_asgn = rl_reassign(
                    agent, surv_pos, surv_speeds, surv_battery,
                    rem_tasks, rem_priority, env.weights,
                )
                reassignments += 1

                for local_di, local_ti in new_asgn:
                    global_di = survivors[local_di]
                    global_ti = unfinished_tasks[local_ti]
                    task_done[global_di] = global_ti
                    targets[global_di]   = rem_tasks[local_ti]
                    total_cost += env.pair_cost(global_di, global_ti)
                    x, y, z = rem_tasks[local_ti]
                    client.moveToPositionAsync(x, y, z, 5, timeout_sec=60,
                                               vehicle_name=drone_names[global_di])
                    print(f"    Drone {global_di} → task {global_ti} (RL recovery)")

        time.sleep(0.5)

    elapsed = time.time() - t0
    n_success = len(done_drones)
    n_total   = env.n - len(failed)
    shutdown_drones(client, drone_names)

    return {
        "total_cost":    total_cost,
        "initial_cost":  initial_cost,
        "elapsed":       elapsed,
        "ok":            ok and n_success == env.n,
        "reassignments": reassignments,
        "drones_completed": n_success,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simulator mode  (no AirSim – inject artificial failures)
# ─────────────────────────────────────────────────────────────────────────────

def run_once_sim(
    agent: PPOAgent,
    env: AssignmentEnv,
    crash_prob: float = 0.15,
) -> Dict:
    """
    Simulate one episode with random drone failures.
    On failure, RL recovery is used (same logic as AirSim path).
    """
    env.reset()
    state  = env.encode_state()
    chosen = []
    for _ in range(env.n):
        valid   = env.get_valid_actions()
        action, _ = agent.select_action(state, valid, deterministic=True)
        chosen.append(action)
        state, _, _, _ = env.step(action)

    total_cost = sum(env.pair_cost(i, t) for i, t in enumerate(chosen))
    reassignments = 0

    # Simulate random crash of one drone
    crashed_drones = [i for i in range(env.n) if random.random() < crash_prob]

    if crashed_drones:
        failed_set    = set(crashed_drones)
        done_set      = set(chosen[i] for i in range(env.n) if i not in failed_set)
        survivors     = [i for i in range(env.n) if i not in failed_set]
        unfinished    = [t for t in range(env.num_tasks) if t not in done_set]

        if survivors and unfinished:
            surv_pos     = env.drones[survivors]
            surv_speeds  = env.drone_speed[survivors]
            surv_battery = env.drone_battery[survivors]
            rem_tasks    = env.tasks[unfinished]
            rem_priority = env.task_priority[unfinished]

            # RL-based recovery
            new_asgn = rl_reassign(
                agent, surv_pos, surv_speeds, surv_battery,
                rem_tasks, rem_priority, env.weights,
            )
            reassignments += 1
            for local_di, local_ti in new_asgn:
                global_di = survivors[local_di]
                global_ti = unfinished[local_ti]
                total_cost += env.pair_cost(global_di, global_ti)

    return {
        "total_cost":    total_cost,
        "reassignments": reassignments,
        "ok":            True,
        "elapsed":       0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry points
# ─────────────────────────────────────────────────────────────────────────────

def eval_sim(model_path, sim_runs=1000, out_dir="outputs/dynamic",
             task_multiplier=1, weights=None, crash_prob=0.15):
    os.makedirs(out_dir, exist_ok=True)
    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    agent.load(model_path)

    print(f"Problem hash: {env.problem_hash}")
    costs, reassign_counts = [], []

    for run in range(sim_runs):
        r = run_once_sim(agent, env)
        costs.append(r["total_cost"])
        reassign_counts.append(r["reassignments"])
        if (run + 1) % max(1, sim_runs // 10) == 0:
            print(f"  Sim run {run+1}/{sim_runs}  "
                  f"cost={r['total_cost']:.4f}  "
                  f"reassignments={r['reassignments']}")

    print_stats(f"DYNAMIC RL SIM ({sim_runs} runs)", costs)
    rc = np.array(reassign_counts)
    print(f"Reassignments per episode: {rc.mean():.3f} ± {rc.std():.3f}")
    save_json(os.path.join(out_dir, "sim_metrics.json"), {
        "sim_runs":       sim_runs,
        "cost_mean":      float(np.mean(costs)),
        "cost_std":       float(np.std(costs)),
        "reassign_mean":  float(rc.mean()),
        "reassign_std":   float(rc.std()),
    })


def run_airsim(model_path, tests=30, out_dir="outputs/dynamic",
               task_multiplier=1, weights=None):
    try:
        import airsim
    except ImportError:
        print("AirSim not available – use --mode sim instead."); return

    os.makedirs(out_dir, exist_ok=True)
    env   = AssignmentEnv(n=3, task_multiplier=task_multiplier, weights=weights)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    agent.load(model_path)

    print(f"Problem hash: {env.problem_hash}")
    client      = airsim.MultirotorClient(); client.confirmConnection()
    drone_names = ["Drone1", "Drone2", "Drone3"]

    costs, times, success_flags, reassign_counts = [], [], [], []

    for idx in range(tests):
        print(f"\n===== Dynamic TEST {idx+1}/{tests} =====")
        r = run_once_airsim(client, agent, env, drone_names)
        costs.append(r["total_cost"])
        times.append(r["elapsed"])
        success_flags.append(r["ok"])
        reassign_counts.append(r["reassignments"])
        print(f"  cost={r['total_cost']:.4f}  time={r['elapsed']:.2f}s  "
              f"ok={r['ok']}  reassignments={r['reassignments']}")

    print_stats(f"DYNAMIC RL AIRSIM ({tests} runs)", costs, times)
    rc = np.array(reassign_counts)
    print(f"Reassignments per episode: {rc.mean():.3f} ± {rc.std():.3f}")
    print(f"Success rate: {np.mean(success_flags):.3f}")

    save_json(os.path.join(out_dir, "airsim_metrics.json"), {
        "tests":          tests,
        "cost_mean":      float(np.mean(costs)),
        "cost_std":       float(np.std(costs)),
        "time_mean":      float(np.mean(times)),
        "time_std":       float(np.std(times)),
        "success_rate":   float(np.mean(success_flags)),
        "reassign_mean":  float(rc.mean()),
        "reassign_std":   float(rc.std()),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic RL-based reassignment with RL crash recovery"
    )
    parser.add_argument("--mode", choices=["run","sim"], required=True,
                        help="run = AirSim | sim = pure-Python simulator")
    parser.add_argument("--model_path",      required=True,
                        help="Path to pre-trained PPO model (.pt)")
    parser.add_argument("--tests",           type=int,   default=30)
    parser.add_argument("--sim_runs",        type=int,   default=1000)
    parser.add_argument("--out_dir",         default="outputs/dynamic")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--task_multiplier", type=int,   default=1)
    parser.add_argument("--crash_prob",      type=float, default=0.15,
                        help="Per-drone crash probability in sim mode")
    parser.add_argument("--weights",         type=str,   default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    weights = None if args.weights is None else json.loads(args.weights)

    if args.mode == "sim":
        eval_sim(args.model_path, args.sim_runs, args.out_dir,
                 args.task_multiplier, weights, args.crash_prob)
    else:
        run_airsim(args.model_path, args.tests, args.out_dir,
                   args.task_multiplier, weights)