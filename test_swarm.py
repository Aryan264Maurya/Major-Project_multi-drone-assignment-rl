import airsim
import time
from AirDrone.assignment_layer import assign_tasks

client = airsim.MultirotorClient()
client.confirmConnection()

# 🔥 RESET SIMULATION (important)
client.reset()
time.sleep(1)

drone_names = ["Drone1", "Drone2", "Drone3"]

# enable + arm
for name in drone_names:
    client.enableApiControl(True, vehicle_name=name)
    client.armDisarm(True, vehicle_name=name)

# -------------------------------
# TAKEOFF
# -------------------------------
for name in drone_names:
    client.takeoffAsync(vehicle_name=name).join()

time.sleep(3)

# -------------------------------
# SPREAD DRONES (avoid collision)
# -------------------------------
offsets = [(0, 0), (5, 0), (-5, 0)]

for i, name in enumerate(drone_names):
    x_offset, y_offset = offsets[i]

    client.moveToPositionAsync(
        x_offset, y_offset, -10,
        5,
        timeout_sec=20,
        vehicle_name=name
    ).join()

time.sleep(2)

# -------------------------------
# STEP 1: Get positions
# -------------------------------
drones = []
for name in drone_names:
    state = client.getMultirotorState(vehicle_name=name)
    pos = state.kinematics_estimated.position
    drones.append((pos.x_val, pos.y_val, pos.z_val))

print("Drone Positions:", drones)

# -------------------------------
# STEP 2: Tasks
# -------------------------------
tasks = [
    (10, 0, -10),
    (0, 20, -10),
    (-10, -10, -10)
]

print("Tasks:", tasks)

# -------------------------------
# STEP 3: Assignment (Hungarian)
# -------------------------------
assignments, cost_matrix = assign_tasks(drones, tasks)

print("Cost Matrix:\n", cost_matrix)

for d, t in assignments:
    print(f"{drone_names[d]} -> {tasks[t]}")

total_cost = sum(cost_matrix[d][t] for d, t in assignments)
print("Total Cost:", total_cost)

# -------------------------------
# STEP 4: Move (FIXED LOGIC)
# -------------------------------
for drone_idx, task_idx in assignments:
    drone_name = drone_names[drone_idx]
    x, y, z = tasks[task_idx]

    print(f"{drone_name} going to {tasks[task_idx]}")

    client.moveToPositionAsync(
        x, y, z,
        5,
        timeout_sec=60,
        vehicle_name=drone_name
    )

# -------------------------------
# WAIT UNTIL REACHED (CRITICAL)
# -------------------------------
def reached(drone_name, target, tol=2):
    state = client.getMultirotorState(vehicle_name=drone_name)
    pos = state.kinematics_estimated.position

    dx = pos.x_val - target[0]
    dy = pos.y_val - target[1]
    dz = pos.z_val - target[2]

    return (dx*dx + dy*dy + dz*dz) ** 0.5 < tol


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

# -------------------------------
# KEEP ALIVE (so you can SEE)
# -------------------------------
time.sleep(5)

# -------------------------------
# LAND (OPTIONAL)
# -------------------------------
for name in drone_names:
    client.landAsync(vehicle_name=name).join()

for name in drone_names:
    client.armDisarm(False, vehicle_name=name)
    client.enableApiControl(False, vehicle_name=name)

print("Done")



# import airsim

# client = airsim.MultirotorClient()
# client.confirmConnection()

# drones = ["Drone1", "Drone2", "Drone3"]

# for drone in drones:
#     client.enableApiControl(True, vehicle_name=drone)
#     client.armDisarm(True, vehicle_name=drone)

# # takeoff
# for drone in drones:
#     client.takeoffAsync(vehicle_name=drone).join()

# # move (Z MUST be negative)
# client.moveToPositionAsync(0, 0, -10, 3, vehicle_name="Drone1")
# client.moveToPositionAsync(5, 5, -10, 3, vehicle_name="Drone2")
# client.moveToPositionAsync(-5, 5, -10, 3, vehicle_name="Drone3")

# # land
# for drone in drones:
#     client.landAsync(vehicle_name=drone).join()

# # disarm
# for drone in drones:
#     client.armDisarm(False, vehicle_name=drone)
#     client.enableApiControl(False, vehicle_name=drone)