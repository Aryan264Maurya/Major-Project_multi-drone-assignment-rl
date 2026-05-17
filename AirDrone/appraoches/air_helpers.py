import json
import os
import time
import numpy as np


def get_airsim_scene(client, drone_names):
    drones = []
    speeds = []
    for name in drone_names:
        state = client.getMultirotorState(vehicle_name=name)
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        drones.append((pos.x_val, pos.y_val, pos.z_val))
        speeds.append((vel.x_val, vel.y_val, vel.z_val))
    return drones, speeds


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
        ok = True
        for i, name in enumerate(drone_names):
            if not reached(client, name, targets[i], tol=tol):
                ok = False
                break
        if ok:
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


def summarize_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci95}


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)