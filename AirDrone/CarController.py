from dataclasses import dataclass
import airsim
import time


@dataclass
class Carinfo():
    speed_a = 0
    speed_l = 0
    
    pos_x = 0
    pos_y = 0
    pos_z = 0
    
    rot_x = 0
    rot_y = 0
    rot_z = 0
    

class CarController():
    def __init__(self):
        # connect to the AirSim simulator
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()  

    def GoForward(self, v):
        self.car_controls.throttle = v
        self.car_controls.steering = 0
        self.client.setCarControls(self.car_controls)

    def GoBackwardStart(self, v):
        self.car_controls.throttle = v
        self.car_controls.is_manual_gear = True
        self.car_controls.manual_gear = -1
        self.car_controls.steering = 0
        self.client.setCarControls(self.car_controls)

    def GoBackwardEnd(self):
        self.car_controls.throttle = 0
        self.car_controls.steering = 0
        self.car_controls.is_manual_gear = False
        self.car_controls.manual_gear = 0
        self.client.setCarControls(self.car_controls)

    def Steer(self, v, steering):
        self.car_controls.throttle = v
        self.car_controls.steering = steering
        self.client.setCarControls(self.car_controls)

    def Stop(self):
        self.car_controls.throttle = 0
        self.car_controls.steering = 0
        self.client.setCarControls(self.car_controls)
        
    def GetCarPose(self):
        position = self.client.simGetVehiclePose().position
        rotation = self.client.simGetVehiclePose().orientation

        speed_a = self.client.getCarState().speed
        speed_l = self.client.getCarState().speed
        
        carinfo = Carinfo()
        
        carinfo.rot_x = rotation.x_val
        carinfo.rot_y = rotation.y_val
        carinfo.rot_z = rotation.z_val
        
        carinfo.pos_x = position.x_val
        carinfo.pos_y = position.y_val
        carinfo.pos_z = position.z_val
        
        carinfo.speed_a = speed_a
        carinfo.speed_l = speed_l
        
        return carinfo
        
        
carcontrol = CarController()