import airsim
import time
import json
from db import log


class Drone:
    def __init__(self, drone_names):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.drone_names = drone_names

        # Enable control for all drones
        for name in drone_names:
            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True, vehicle_name=name)

    # -------------------------------
    # TAKEOFF
    # -------------------------------
    def TakeOff(self):
        for name in self.drone_names:
            self.client.takeoffAsync(vehicle_name=name).join()
            print(f"{name} Takeoff")
            log(name, "takeoff")

    # -------------------------------
    # LAND
    # -------------------------------
    def Landed(self):
        for name in self.drone_names:
            self.client.landAsync(vehicle_name=name).join()
            print(f"{name} Landing")
            log(name, "land")

        # disarm after landing
        for name in self.drone_names:
            self.client.armDisarm(False, vehicle_name=name)
            self.client.enableApiControl(False, vehicle_name=name)

    # -------------------------------
    # GET CURRENT POSITIONS
    # -------------------------------
    def get_drone_positions(self):
        drones = []

        for name in self.drone_names:
            state = self.client.getMultirotorState(vehicle_name=name)
            pos = state.kinematics_estimated.position

            drones.append((pos.x_val, pos.y_val, pos.z_val))

        return drones

    # -------------------------------
    # MOVE USING ASSIGNMENT (NEW 🔥)
    # -------------------------------
    def move_drones_to_tasks(self, assignments, tasks):
        futures = []

        for drone_idx, task_idx in assignments:
            drone_name = self.drone_names[drone_idx]
            x, y, z = tasks[task_idx]

            print(f"{drone_name} moving to {x, y, z}")

            f = self.client.moveToPositionAsync(
                x, y, z,
                velocity=5,
                vehicle_name=drone_name
            )

            futures.append((f, drone_name, x, y, z))

        # wait + log movement
        for f, drone_name, x, y, z in futures:
            f.join()
            log(drone_name, "move", x, y, z)

    # -------------------------------
    # OLD ROUTE REPLAY (UNCHANGED)
    # -------------------------------
    def ReplayRoute(self):
        with open('Route.json', 'r', encoding='utf-8') as file:
            data = json.load(file)

        location_x = data.get("Location.X", [])
        location_y = data.get("Location.Y", [])
        location_z = data.get("Location.Z", [])

        # collect frames
        all_frames = set()
        for arr in [location_x, location_y, location_z]:
            all_frames.update([item["frame"] for item in arr])
        all_frames = sorted(all_frames)

        interp_x = interpolate_axis(location_x, all_frames)
        interp_y = interpolate_axis(location_y, all_frames)
        interp_z = interpolate_axis(location_z, all_frames)

        for i in range(len(all_frames) - 1):
            position = airsim.Vector3r(
                interp_x[i] / 100,
                interp_y[i] / 100,
                -interp_z[i] / 100
            )

            print("Moving to:", position)

            self.client.moveToPositionAsync(
                position.x_val,
                position.y_val,
                position.z_val,
                3
            ).join()

            log("Drone1", "move", position.x_val, position.y_val, position.z_val)

            time.sleep(0.1)


# -------------------------------
# HELPER FUNCTION
# -------------------------------
def interpolate_axis(axis_list, target_frames):
    if not axis_list:
        return [0.0] * len(target_frames)

    axis_list = sorted(axis_list, key=lambda x: x["frame"])
    frames = [item["frame"] for item in axis_list]
    values = [item["value"] for item in axis_list]

    result = []
    for f in target_frames:
        if f <= frames[0]:
            result.append(values[0])
        elif f >= frames[-1]:
            result.append(values[-1])
        else:
            for i in range(len(frames) - 1):
                if frames[i] <= f <= frames[i + 1]:
                    t = (f - frames[i]) / (frames[i + 1] - frames[i])
                    v = values[i] + t * (values[i + 1] - values[i])
                    result.append(v)
                    break
    return result