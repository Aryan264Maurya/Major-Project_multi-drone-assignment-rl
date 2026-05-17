"""
baseline_hungarian.py
---------------------
Hungarian-algorithm baseline for drone task assignment.

Supports AirSim (default) and the pure-Python simulator (--use_sim).
Reports mean ± std, 95 % CI, and a one-sample t-test over --runs episodes.

Usage
-----
    # AirSim, 30 runs (default)
    python baseline_hungarian.py --runs 30

    # Simulator only (no AirSim), 1000 runs for policy-transfer analysis
    python baseline_hungarian.py --runs 1000 --use_sim

    # Ablate cost weights on 20 random scenes, then run 30 simulator tests
    python baseline_hungarian.py --ablate --use_sim --runs 30
"""

import argparse
import time
import random
import numpy as np
from scipy import stats

from assignment_layer import assign_tasks, run_weight_ablation, run_weight_grid_search
from assignment_env   import AssignmentEnv
from drone_simulator  import make_client        # noqa: F401  (AirSim / sim switch)


# ─────────────────────────────────────────────────────────────────────────────
# Shared flight helpers  (imported by dqn_assignment and ppo_assignment)
# ─────────────────────────────────────────────────────────────────────────────

def get_positions(client, drone_names):
    positions = []
    for name in drone_names:
        s   = client.getMultirotorState(vehicle_name=name)
        pos = s.kinematics_estimated.position
        positions.append((pos.x_val, pos.y_val, pos.z_val))
    return positions


def reached(client, drone_name, target, tol=2.5):
    s   = client.getMultirotorState(vehicle_name=drone_name)
    pos = s.kinematics_estimated.position
    dx  = pos.x_val - target[0]
    dy  = pos.y_val - target[1]
    dz  = pos.z_val - target[2]
    return (dx*dx + dy*dy + dz*dz) ** 0.5 < tol


def wait_all(client, drone_names, targets, max_wait=90.0, tol=2.5, dt=0.25):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if all(reached(client, n, targets[i], tol) for i, n in enumerate(drone_names)):
            return True
        time.sleep(dt)
    return False


def prepare_drones(client, drone_names, z_cruise=-10.0):
    client.reset()
    time.sleep(0.5)
    for name in drone_names:
        client.enableApiControl(True,  vehicle_name=name)
        client.armDisarm(True,         vehicle_name=name)
    for name in drone_names:
        client.takeoffAsync(vehicle_name=name).join()
    time.sleep(0.5)

    offsets = [(0.0, 0.0), (5.0, 0.0), (-5.0, 0.0)]
    targets = [(offsets[i][0], offsets[i][1], z_cruise) for i in range(len(drone_names))]
    for i, name in enumerate(drone_names):
        client.moveToPositionAsync(*targets[i], 5, timeout_sec=20, vehicle_name=name)
    wait_all(client, drone_names, targets, max_wait=40)
    time.sleep(0.5)


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


# ─────────────────────────────────────────────────────────────────────────────
# Single AirSim test run
# ─────────────────────────────────────────────────────────────────────────────

def run_once_airsim(client, drone_names, task_coords, z_level=-10.0):
    prepare_drones(client, drone_names, z_cruise=z_level)
    drones = get_positions(client, drone_names)

    n = len(drone_names);  m = len(task_coords)
    drone_battery = np.ones(n,  dtype=np.float32)
    drone_speed   = np.ones(n,  dtype=np.float32)
    task_priority = np.random.randint(1, 6, size=m).astype(np.float32)

    assignments, cost_matrix = assign_tasks(
        drones, task_coords,
        drone_battery=drone_battery, task_priority=task_priority,
        drone_speed=drone_speed,
    )
    total_cost = float(sum(cost_matrix[d][t] for d, t in assignments))

    print(f"  Cost Matrix:\n{np.round(cost_matrix, 3)}")
    for d, t in assignments:
        print(f"  {drone_names[d]} -> Task {t} {task_coords[t]}")
    print(f"  Total cost: {total_cost:.4f}")

    for di, ti in assignments:
        x, y, z = task_coords[ti]
        client.moveToPositionAsync(x, y, z, 5, timeout_sec=60, vehicle_name=drone_names[di])

    sorted_asgn = sorted(assignments, key=lambda a: a[0])
    targets = [task_coords[ti] for _, ti in sorted_asgn]

    t0      = time.time()
    success = wait_all(client, drone_names, targets, max_wait=90)
    elapsed = time.time() - t0

    shutdown_drones(client, drone_names)
    return total_cost, elapsed, success


# ─────────────────────────────────────────────────────────────────────────────
# Single simulator test run  (no AirSim)
# ─────────────────────────────────────────────────────────────────────────────

def run_once_sim(env: AssignmentEnv) -> dict:
    """One run in the pure-Python simulator."""
    env.reset()
    assignments, cost_matrix = assign_tasks(
        env.drones, env.tasks,
        drone_battery=env.drone_battery,
        task_priority=env.task_priority,
        drone_speed=env.drone_speed,
        weights=env.weights,
    )
    total_cost = float(sum(cost_matrix[d][t] for d, t in assignments))
    return {"cost": total_cost, "success": True, "elapsed": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Statistical summary helper
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(label: str, costs: np.ndarray, times: np.ndarray):
    n       = len(costs)
    se      = costs.std() / np.sqrt(n)
    ci_lo   = costs.mean() - 1.96 * se
    ci_hi   = costs.mean() + 1.96 * se
    t_stat, p_val = stats.ttest_1samp(costs, 0.0)

    print(f"\n========== {label} SUMMARY ==========")
    print(f"Runs              : {n}")
    print(f"Cost  mean ± std  : {costs.mean():.4f} ± {costs.std():.4f}")
    print(f"Cost  95 % CI     : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Cost  [min, max]  : [{costs.min():.4f}, {costs.max():.4f}]")
    if times is not None and len(times):
        print(f"Time  mean ± std  : {times.mean():.2f} ± {times.std():.2f} s")
    print(f"t-test vs 0       : t={t_stat:.3f}, p={p_val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hungarian baseline evaluation")
    parser.add_argument("--runs",       type=int,  default=30,
                        help="Number of independent test runs (default 30)")
    parser.add_argument("--use_sim",    action="store_true",
                        help="Pure-Python simulator – no AirSim required")
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--ablate",     action="store_true",
                        help="Run weight ablation before tests")
    parser.add_argument("--grid_search",action="store_true",
                        help="Run exhaustive weight grid search (slow)")
    args = parser.parse_args()

    random.seed(args.seed);  np.random.seed(args.seed)

    drone_names = ["Drone1", "Drone2", "Drone3"]
    task_coords = [
        ( 10.0,   0.0, -10.0),
        (  0.0,  20.0, -10.0),
        (-10.0, -10.0, -10.0),
    ]

    # ── Optional weight ablation ────────────────────────────────────────────
    if args.ablate or args.grid_search:
        env_tmp = AssignmentEnv(n=3)
        scenes  = []
        for _ in range(20):
            env_tmp.reset()
            scenes.append({
                "drones":        env_tmp.drones.tolist(),
                "tasks":         env_tmp.tasks.tolist(),
                "drone_battery": env_tmp.drone_battery.tolist(),
                "task_priority": env_tmp.task_priority.tolist(),
                "drone_speed":   env_tmp.drone_speed.tolist(),
            })

        if args.ablate:
            print("Running one-at-a-time weight ablation on 20 random scenes …")
            run_weight_ablation(scenes)

        if args.grid_search:
            print("Running weight grid search on 20 random scenes …")
            run_weight_grid_search(scenes)

    # ── Test runs ───────────────────────────────────────────────────────────
    costs, times = [], []

    if args.use_sim:
        env = AssignmentEnv(n=3)
        print(f"\nRunning {args.runs} simulator runs …")
        for run in range(args.runs):
            r = run_once_sim(env)
            costs.append(r["cost"])
            times.append(r["elapsed"])
            if (run + 1) % max(1, args.runs // 10) == 0:
                print(f"  Run {run + 1}/{args.runs}  cost={r['cost']:.4f}")
    else:
        from drone_simulator import make_client
        client = make_client(use_sim=False, drone_names=drone_names)
        client.confirmConnection()
        print(f"\nRunning {args.runs} AirSim tests …")
        for run in range(args.runs):
            print(f"\n--- Run {run + 1}/{args.runs} ---")
            cost, elapsed, ok = run_once_airsim(client, drone_names, task_coords)
            costs.append(cost)
            times.append(elapsed)
            print(f"  Time: {elapsed:.2f}s | {'OK' if ok else 'TIMEOUT'}")

    print_summary("HUNGARIAN BASELINE", np.array(costs), np.array(times))


if __name__ == "__main__":
    main()