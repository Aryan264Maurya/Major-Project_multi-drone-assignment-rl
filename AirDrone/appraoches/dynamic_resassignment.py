import time
import numpy as np
import airsim

from assignment_layer import assign_tasks


def get_drone_state(client, name):
    state = client.getMultirotorState(vehicle_name=name)
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    col = client.simGetCollisionInfo(vehicle_name=name)
    return (
        (pos.x_val, pos.y_val, pos.z_val),
        (vel.x_val, vel.y_val, vel.z_val),
        bool(col.has_collided),
    )


def reached(client, drone_name, target, tol=2.5):
    state = client.getMultirotorState(vehicle_name=drone_name)
    pos = state.kinematics_estimated.position
    dx = pos.x_val - target[0]
    dy = pos.y_val - target[1]
    dz = pos.z_val - target[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5 < tol


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

    start = time.time()
    while time.time() - start < 40:
        ok = True
        for i, name in enumerate(drone_names):
            if not reached(client, name, targets[i]):
                ok = False
                break
        if ok:
            break
        time.sleep(0.25)

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


def dynamic_assign_and_execute(client, drone_names, tasks, drone_battery=None, task_priority=None, max_wait=90):
    active = drone_names[:]
    pending = list(range(len(tasks)))
    completed = set()

    if drone_battery is None:
        drone_battery = {name: 1.0 for name in drone_names}

    if task_priority is None:
        task_priority = np.random.randint(1, 6, size=len(tasks)).astype(np.float32)

    all_assignments = []

    while len(active) > 0 and len(pending) > 0:
        active_positions = []
        active_speeds = []
        active_names = []

        alive_now = []
        for name in active:
            pos, vel, collided = get_drone_state(client, name)
            if collided:
                print(f"[CRASH] {name} collided. Removing from active set.")
                continue
            active_positions.append(pos)
            active_speeds.append(vel)
            active_names.append(name)
            alive_now.append(name)

        active = alive_now

        if len(active) == 0 or len(pending) == 0:
            break

        pending_tasks = [tasks[i] for i in pending]
        pending_priorities = [task_priority[i] for i in pending]
        pending_battery = [drone_battery[name] for name in active_names]

        assignments, cost_matrix = assign_tasks(
            active_positions,
            pending_tasks,
            drone_battery=pending_battery,
            task_priority=pending_priorities,
            drone_speed=active_speeds,
        )

        if len(assignments) == 0:
            break

        print("\nNew assignment batch:")
        batch_targets = {}
        batch_task_global = {}

        for d_idx, t_idx in assignments:
            drone_name = active_names[d_idx]
            task_global_idx = pending[t_idx]
            batch_targets[drone_name] = tasks[task_global_idx]
            batch_task_global[drone_name] = task_global_idx
            print(f"{drone_name} -> Task {task_global_idx} {tasks[task_global_idx]}")
            client.moveToPositionAsync(
                tasks[task_global_idx][0],
                tasks[task_global_idx][1],
                tasks[task_global_idx][2],
                5,
                timeout_sec=60,
                vehicle_name=drone_name
            )

        start = time.time()
        while time.time() - start < max_wait:
            changed = False

            for drone_name, task_idx in list(batch_task_global.items()):
                _, _, collided = get_drone_state(client, drone_name)

                if collided:
                    print(f"[CRASH] {drone_name} crashed while flying.")
                    if drone_name in active:
                        active.remove(drone_name)
                    changed = True
                    continue

                if reached(client, drone_name, tasks[task_idx], tol=2.5):
                    if task_idx not in completed:
                        completed.add(task_idx)
                        if task_idx in pending:
                            pending.remove(task_idx)
                        all_assignments.append((drone_name, task_idx))
                        print(f"[DONE] {drone_name} completed Task {task_idx}")
                        changed = True

            if changed:
                break

            if len(pending) == 0:
                break

            time.sleep(0.25)

        if len(pending) == 0:
            break

    return all_assignments, completed


if __name__ == "__main__":
    client = airsim.MultirotorClient()
    client.confirmConnection()

    drone_names = ["Drone1", "Drone2", "Drone3"]
    tasks = [
        (10, 0, -10),
        (0, 20, -10),
        (-10, -10, -10),
        (15, 10, -10),
        (-15, 5, -10),
        (5, -15, -10),
    ]

    prepare_drones(client, drone_names)

    assignments, completed = dynamic_assign_and_execute(
        client=client,
        drone_names=drone_names,
        tasks=tasks,
    )

    print("\nFinal assignments:", assignments)
    print("Completed task indices:", sorted(list(completed)))

    time.sleep(3)
    shutdown_drones(client, drone_names)
    print("Done")