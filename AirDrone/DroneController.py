from multiprocessing import Value
import airsim
import time
import math
import keyboard
import json

class DroneInfo():
    gps_pos_altitude = 0
    gps_pos_latitude = 0
    gps_pos_longitude = 0

    ori_w = 0
    ori_x = 0
    ori_y = 0
    ori_z = 0

    # linear_velocity
    l_v_x = 0
    l_v_y = 0
    l_v_z = 0

    # angular_velocity
    a_v_x = 0
    a_v_y = 0
    a_v_z = 0

    roll = 0
    yaw = 0
    pitch = 0

    linear_speed = 0


class Drone():
    def __init__(self):
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def TakeOff(self):
        if airsim.LandedState.Landed == 0:
            self.client.takeoffAsync().join()
            print("take off")
        else:
            self.client.hoverAsync().join()
            print("already flying")

    def Landed(self):
        if airsim.LandedState.Landed == 1:
            print("already landed")
        else:
            self.client.landAsync().join()
            print("now landing")

    def DroneMoveByTime(self, vx, vy, vz, duration=3):
        print("now speed", vx, vy, vz)
        self.client.moveByVelocityAsync(vx, vy, vz, duration)
        time.sleep(duration)
        print("End Moving")

    def GetState(self):
        # State = self.client.getMultirotorState()
        State = DroneInfo()
        gps_pos = self.client.getMultirotorState().gps_location
        State.gps_pos_altitude = gps_pos.altitude
        State.gps_pos_longitude = gps_pos.longitude
        State.gps_pos_latitude = gps_pos.latitude

        orientation = self.client.getMultirotorState().kinematics_estimated.orientation
        State.ori_w = orientation.w_val
        State.ori_x = orientation.x_val
        State.ori_y = orientation.y_val
        State.ori_z = orientation.z_val

        a_velocity = self.client.getMultirotorState().kinematics_estimated.angular_velocity
        State.a_v_x = a_velocity.x_val
        State.a_v_y = a_velocity.y_val
        State.a_v_z = a_velocity.z_val

        l_velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        State.l_v_x = l_velocity.x_val
        State.l_v_y = l_velocity.y_val
        State.l_v_z = l_velocity.z_val

        State.roll = self.client.getMultirotorState().rc_data.roll
        State.yaw = self.client.getMultirotorState().rc_data.yaw
        State.pitch = self.client.getMultirotorState().rc_data.pitch

        State.linear_speed = math.sqrt(pow(State.l_v_x,2)+pow(State.l_v_y,2)+pow(State.l_v_z,2))

        return State

    def Hover(self):
        landed = self.client.getMultirotorState()

    def MoveToPosition(self,x,y,z,duration=3):
        self.client.moveToPositionAsync(x, y, z, duration).join()

    def Reset(self):
        self.client.reset()

    def SnowTeleport(self):
        position = airsim.Vector3r(9.87236677 , -279.41862836, -22.81082173) # snow XYZ 0.01 -Z
        heading = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(position, heading)
        self.client.simSetVehiclePose(pose, True)

    def OvercastTeleport(self):
        position = airsim.Vector3r(213.24547002 , -429.66393833, 19.260)
        heading = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(position, heading)
        self.client.simSetVehiclePose(pose, True)    

    def RainTeleport(self):
        position = airsim.Vector3r(-315.200 , -444.920, 12.520)
        heading = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(position, heading)
        self.client.simSetVehiclePose(pose, True)
    
    def SandTeleport(self):
        position = airsim.Vector3r(-335.50020795 , 282.16650494, 19.50313758)
        heading = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(position, heading)
        self.client.simSetVehiclePose(pose, True)

    def ClearTeleport(self):
        position = airsim.Vector3r(-152.28332052 , 236.00371837, 66.40114393)
        heading = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(position, heading)
        self.client.simSetVehiclePose(pose, True)

    def FogTeleport(self):
        position = airsim.Vector3r(-430.39138328 , -91.7420777, 13.86624983)
        heading = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(position, heading)
        self.client.simSetVehiclePose(pose, True)


    def CloseAPI(self):
        self.client.enableApiControl(False)

    def KeyboardControl(self,x):
        w = keyboard.KeyboardEvent('down', 150, 'w')             # forward
        s = keyboard.KeyboardEvent('down', 150, 's')             # back
        a = keyboard.KeyboardEvent('down', 150, 'a')             # left
        d = keyboard.KeyboardEvent('down', 150, 'd')             # right
        up = keyboard.KeyboardEvent('down', 150, 'up')           # up
        down = keyboard.KeyboardEvent('down', 150, 'down')       # down
        left = keyboard.KeyboardEvent('down', 150, 'left')       # left
        right = keyboard.KeyboardEvent('down', 150, 'right')     # right
        k = keyboard.KeyboardEvent('down', 28, 'k')             # get control
        l = keyboard.KeyboardEvent('down', 28, 'l')             # release control
        if x.event_type == 'down' and x.name == w.name:
            self.client.moveByVelocityBodyFrameAsync(3, 0, 0, 0.5)
            print("forward")
        elif x.event_type == 'down' and x.name == s.name:
            self.client.moveByVelocityBodyFrameAsync(-3, 0, 0, 0.5)
            print("back")
        elif x.event_type == 'down' and x.name == a.name:
            self.client.moveByVelocityBodyFrameAsync(0, -2, 0, 0.5)
            print("left")
        elif x.event_type == 'down' and x.name == d.name:
            self.client.moveByVelocityBodyFrameAsync(0, 2, 0, 0.5)
            print("right")
        elif x.event_type == 'down' and x.name == up.name:
            self.client.moveByVelocityBodyFrameAsync(0, 0, -0.5, 0.5)
            print("up")
        elif x.event_type == 'down' and x.name == down.name:
            self.client.moveByVelocityBodyFrameAsync(0, 0, 0.5, 0.5)
            print("down")
        elif x.event_type == 'down' and x.name == left.name:
            self.client.rotateByYawRateAsync(-20, 0.5)
            print("turn left")
        elif x.event_type == 'down' and x.name == right.name:
            self.client.rotateByYawRateAsync(20, 0.5)
            print("turn right")
        elif x.event_type == 'down' and x.name == k.name:
            # get control
            self.client.enableApiControl(True)
            print("get control")
            # unlock
            self.client.armDisarm(True)
            print("unlock")
            # Async methods returns Future. Call join() to wait for task to complete.
            self.client.takeoffAsync().join()
            print("takeoff")
        elif x.event_type == 'down' and x.name == l.name:
            keyboard.wait("l")
            keyboard.unhook_all()
        else:
            self.client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.5).join()
            self.client.hoverAsync().join()
            print("hovering")

    def ReplayRoute(self):
        with open('Route.json', 'r', encoding='utf-8') as file:
            data = json.load(file)

        location_x = data.get("Location.X", [])
        location_y = data.get("Location.Y", [])
        location_z = data.get("Location.Z", [])

        # lerp
        all_frames = set()
        for arr in [location_x, location_y, location_z]:
            all_frames.update([item["frame"] for item in arr])
        all_frames = sorted(all_frames)

        interp_x = interpolate_axis(location_x, all_frames)
        interp_y = interpolate_axis(location_y, all_frames)
        interp_z = interpolate_axis(location_z, all_frames)

        for i in range(len(all_frames) - 1):
            position = airsim.Vector3r(interp_x[i]/100, interp_y[i]/100, -interp_z[i]/100)
            position_next = airsim.Vector3r(interp_x[i+1]/100, interp_y[i+1]/100, -interp_z[i+1]/100)
            time = (all_frames[i+1] - all_frames[i]) / 30  # 30 FPS
            speed = Calc_distance(position, position_next) / time if time > 0 else 1
            print(position)
            self.client.moveToPositionAsync(position.x_val, position.y_val, position.z_val, speed).join()


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
                if frames[i] <= f <= frames[i+1]:
                    t = (f - frames[i]) / (frames[i+1] - frames[i])
                    v = values[i] + t * (values[i+1] - values[i])
                    result.append(v)
                    break
    return result


def Calc_distance(pos1, pos2):
    return math.sqrt((pos1.x_val - pos2.x_val) ** 2 + (pos1.y_val - pos2.y_val) ** 2 + (pos1.z_val - pos2.z_val) ** 2)

if __name__ == '__main__':
    dronne = Drone()
    # add sleep to make sure excute at same time
    dronne.ReplayRoute()

dronecontrol = Drone()

