import airsim
import time
from assignment_layer import assign_tasks

client = airsim.MultirotorClient()
client.confirmConnection()

client.reset()
time.sleep(1)

drone_names = ["Drone1", "Drone2", "Drone3"]

for name in drone_names:
    client.enableApiControl(True, vehicle_name=name)
    client.armDisarm(True, vehicle_name=name)

for name in drone_names:
    client.takeoffAsync(vehicle_name=name).join()

time.sleep(2)

offsets = [(0, 0), (5, 0), (-5, 0)]

move_handles = []
for i, name in enumerate(drone_names):
    x_offset, y_offset = offsets[i]
    handle = client.moveToPositionAsync(
        x_offset, y_offset, -10,
        5,
        timeout_sec=20,
        vehicle_name=name
    )
    move_handles.append(handle)

for h in move_handles:
    h.join()

time.sleep(1)

drones = []
for name in drone_names:
    state = client.getMultirotorState(vehicle_name=name)
    pos = state.kinematics_estimated.position
    drones.append((pos.x_val, pos.y_val, pos.z_val))

print("Drone Positions:", drones)

tasks = [
    (10, 0, -10),
    (0, 20, -10),
    (-10, -10, -10)
]

print("Tasks:", tasks)

assignments, cost_matrix = assign_tasks(drones, tasks)

print("\nCost Matrix:\n", cost_matrix)

for d, t in assignments:
    print(f"{drone_names[d]} -> {tasks[t]}")

total_cost = sum(cost_matrix[d][t] for d, t in assignments)
print("Total Cost:", total_cost)

move_handles = []
for drone_idx, task_idx in assignments:
    drone_name = drone_names[drone_idx]
    x, y, z = tasks[task_idx]

    print(f"{drone_name} going to {tasks[task_idx]}")

    handle = client.moveToPositionAsync(
        x, y, z,
        5,
        timeout_sec=60,
        vehicle_name=drone_name
    )
    move_handles.append((drone_name, handle))

for _, handle in move_handles:
    handle.join()

def reached(drone_name, target, tol=2):
    state = client.getMultirotorState(vehicle_name=drone_name)
    pos = state.kinematics_estimated.position
    dx = pos.x_val - target[0]
    dy = pos.y_val - target[1]
    dz = pos.z_val - target[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5 < tol

print("\nWaiting for drones to reach targets...\n")

while True:
    all_reached = True
    for drone_idx, task_idx in assignments:
        drone_name = drone_names[drone_idx]
        target = tasks[task_idx]
        if not reached(drone_name, target):
            all_reached = False
    if all_reached:
        break
    time.sleep(1)

print("All drones reached targets!")
time.sleep(5)

for name in drone_names:
    client.landAsync(vehicle_name=name).join()

for name in drone_names:
    client.armDisarm(False, vehicle_name=name)
    client.enableApiControl(False, vehicle_name=name)

print("Done")