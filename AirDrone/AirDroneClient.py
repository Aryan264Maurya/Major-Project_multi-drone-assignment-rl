import sys
from dataclasses import dataclass

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import keyboard

import DroneController  # 假设只需要无人机控制

# mobile configuration
@dataclass
class Vehicle():
    acc = 1
    vehicle_type = 'drone'  # 设置为无人机模式

veh = Vehicle()

# form application #
app = QApplication(sys.argv)

class AirDroneClientWindow(QWidget):
    button_takeoff = QPushButton()
    button_land = QPushButton()
    button_up = QPushButton()
    button_down = QPushButton()
    
    button_f = QPushButton()
    button_b = QPushButton()
    button_l = QPushButton()
    button_r = QPushButton()
    
    button_weather = QPushButton()
    button_task = QPushButton()
    button_algorithm = QPushButton()
    button_formation = QPushButton()
    button_keyboard = QPushButton()
    
    weather_selection = QComboBox()
    
    label_gps_alt = QLabel()
    label_gps_lat = QLabel()
    label_gps_lon = QLabel()
    label_rot_w = QLabel()
    label_rot_y = QLabel()
    label_rot_z = QLabel()
    label_speed = QLabel()
    label_angle_speed = QLabel()

    refreshinfo = QTimer()

    def initUI(self):
        self.resize(500, 400)  # 调整为更合适的尺寸
        self.setWindowTitle("U21Data-2 Client")

        #self.set_background()

        self.setObjectName("AirDroneClientWindow")  # 设置对象名称匹配选择器
    

        self.setStyleSheet(self.styleSheet())

        # 主垂直布局
        main_layout = QVBoxLayout()
        
        # 添加Start Point部分
        start_group = QGroupBox("Start Point")
        start_layout = QHBoxLayout()

        self.start_combo = QComboBox()
        self.start_combo.setEditable(True)
        self.start_combo.lineEdit().setReadOnly(True)
        self.start_combo.lineEdit().setAlignment(Qt.AlignCenter)
        self.start_combo.addItems(["9.8, -279, -22", "213, -429, 19","-315, -444, 12", "-335, 282, 19", "-152, 236, 66", "-430, -91, 13"])
        self.start_combo.setCurrentIndex(-1)
        self.start_combo.lineEdit().setPlaceholderText("StartPoint")
    

        self.weather_selection = QComboBox()
        self.weather_selection.setEditable(True)
        self.weather_selection.lineEdit().setReadOnly(True)
        self.weather_selection.lineEdit().setAlignment(Qt.AlignCenter)
        self.weather_selection.addItems(["Clear", "Rain", "Snow", "Fog", "Overcast","Sand"])
        self.weather_selection.setCurrentIndex(-1)
        self.weather_selection.lineEdit().setPlaceholderText("Weather")
    
        for widget in [self.start_combo, self.weather_selection]:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    
        start_layout.addWidget(self.start_combo)
        start_layout.addWidget(self.weather_selection)
        start_group.setLayout(start_layout)
        
        # 添加功能按钮部分
        func_group = QGroupBox("Functions")
        func_layout = QHBoxLayout()
    
        self.button_keyboard = QComboBox()
        self.button_keyboard.setEditable(True)
        self.button_keyboard.lineEdit().setReadOnly(True)
        self.button_keyboard.lineEdit().setAlignment(Qt.AlignCenter)
        self.button_keyboard.addItems(["Yes", "No"])
        self.button_keyboard.setCurrentIndex(-1)
        self.button_keyboard.lineEdit().setPlaceholderText("Task")



        #Task
        self.task_combo = QComboBox()
        self.task_combo.setEditable(True)  # 设置为可编辑
        self.task_combo.lineEdit().setReadOnly(True)  # 设置行编辑为只读
        self.task_combo.lineEdit().setAlignment(Qt.AlignCenter)  # 文字居中
        self.task_combo.addItems(["Task 1", "Task 2", "Task 3"])
        self.task_combo.setCurrentIndex(-1)  # 不选中任何项
        self.task_combo.lineEdit().setPlaceholderText("Task")

    
        # 设置Algorithm下拉框选项
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.setEditable(True)  # 设置为可编辑
        self.algorithm_combo.lineEdit().setReadOnly(True)  # 设置行编辑为只读
        self.algorithm_combo.lineEdit().setAlignment(Qt.AlignCenter)  # 文字居中
        self.algorithm_combo.addItems(["Algorithm 1", "Algorithm 2", "Algorithm 3"])
        self.algorithm_combo.setCurrentIndex(-1)  # 不选中任何项
        self.algorithm_combo.lineEdit().setPlaceholderText("Algorithm")  # 设置占位文本

        # 设置Formation下拉框选项
        self.formation_combo = QComboBox()
        self.formation_combo.setEditable(True)
        self.formation_combo.lineEdit().setReadOnly(True)
        self.formation_combo.lineEdit().setAlignment(Qt.AlignCenter)
        self.formation_combo.addItems(["Line Formation", "V Formation", "Square Formation", "Circle Formation"])
        self.formation_combo.setCurrentIndex(-1)
        self.formation_combo.lineEdit().setPlaceholderText("Formation")
    
        # 设置控件大小策略
        for widget in [self.task_combo, self.algorithm_combo, 
                       self.formation_combo, self.button_keyboard]:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    
        # 添加控件到布局
        func_layout.addWidget(self.task_combo)
        func_layout.addWidget(self.algorithm_combo)
        func_layout.addWidget(self.formation_combo)
        func_layout.addWidget(self.button_keyboard)
        func_group.setLayout(func_layout)
        
        # 添加飞行控制部分
        flight_group = QGroupBox("Flight Controls")
        flight_layout = QVBoxLayout()

        # 将Takeoff, Land, Up, Down四个按钮放在同一行
        top_row_layout = QHBoxLayout()
        self.button_takeoff = QPushButton("Takeoff")
        self.button_land = QPushButton("Land")
        self.button_up = QPushButton("Up")
        self.button_down = QPushButton("Down")

        # 设置按钮大小策略，使它们均匀分布
        for button in [self.button_takeoff, self.button_land, 
                       self.button_up, self.button_down]:
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        top_row_layout.addWidget(self.button_takeoff)
        top_row_layout.addWidget(self.button_land)
        top_row_layout.addWidget(self.button_up)
        top_row_layout.addWidget(self.button_down)
        top_row_layout.setSpacing(5)  # 设置按钮间距
        
        # 前后左右
        move_layout = QGridLayout()
        self.button_f = QPushButton("Move Forward")
        self.button_b = QPushButton("Move Backward")
        self.button_l = QPushButton("Move Left")
        self.button_r = QPushButton("Move Right")
        move_layout.addWidget(self.button_f, 0, 1)
        move_layout.addWidget(self.button_b, 2, 1)
        move_layout.addWidget(self.button_l, 1, 0)
        move_layout.addWidget(self.button_r, 1, 2)

        flight_layout.addLayout(top_row_layout)
        flight_layout.addLayout(move_layout)
        flight_group.setLayout(flight_layout)
        
        # 添加日志信息部分
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.label_gps_alt = QLabel("GPS Altitude: ")
        self.label_gps_lat = QLabel("GPS Latitude: ")
        self.label_gps_lon = QLabel("GPS Longitude: ")
        self.label_rot_w = QLabel("Rotation w: ")
        self.label_rot_y = QLabel("Rotation y: ")
        self.label_rot_z = QLabel("Rotation z: ")
        self.label_speed = QLabel("Linear Speed: ")
        self.label_angle_speed = QLabel("Angle Speed: ")
        
        log_layout.addWidget(self.label_gps_alt)
        log_layout.addWidget(self.label_gps_lat)
        log_layout.addWidget(self.label_gps_lon)
        log_layout.addWidget(self.label_rot_w)
        log_layout.addWidget(self.label_rot_y)
        log_layout.addWidget(self.label_rot_z)
        log_layout.addWidget(self.label_speed)
        log_layout.addWidget(self.label_angle_speed)
        log_group.setLayout(log_layout)
        
        # 将所有组添加到主布局
        main_layout.addWidget(start_group)
        main_layout.addWidget(func_group)
        main_layout.addWidget(flight_group)
        main_layout.addWidget(log_group)
        
        self.setLayout(main_layout)
        
        # 连接信号槽
        self.button_takeoff.clicked.connect(DroneController.dronecontrol.TakeOff)
        self.button_land.clicked.connect(DroneController.dronecontrol.Landed)
        self.button_up.clicked.connect(self.up)
        self.button_down.clicked.connect(self.down)
        self.button_f.pressed.connect(self.forward)
        self.button_b.pressed.connect(self.backward)
        self.button_l.pressed.connect(self.left)
        self.button_r.pressed.connect(self.right)
        self.button_f.released.connect(self.stop)
        self.button_b.released.connect(self.stop)
        self.button_l.released.connect(self.stop)
        self.button_r.released.connect(self.stop)
        self.button_keyboard.currentIndexChanged.connect(self.EnableKeyboard)
        self.weather_selection.currentIndexChanged.connect(self.change_weather)
        self.algorithm_combo.currentIndexChanged.connect(self.on_algorithm_changed)
        self.formation_combo.currentIndexChanged.connect(self.on_formation_changed)
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)
        self.start_combo.currentIndexChanged.connect(self.on_start_changed)
        
        # 自动刷新信息
        self.refreshinfo.start(1000)
        self.refreshinfo.timeout.connect(self.setInfo)

        self.show()

    def setInfo(self):
        self.label_gps_alt.setText("GPS Altitude: " + str(DroneController.dronecontrol.GetState().gps_pos_altitude))
        self.label_gps_lat.setText("GPS Latitude: " + str(DroneController.dronecontrol.GetState().gps_pos_latitude))
        self.label_gps_lon.setText("GPS Longitude: " + str(DroneController.dronecontrol.GetState().gps_pos_longitude))
        
        self.label_rot_w.setText("Rotation w: " + str(DroneController.dronecontrol.GetState().ori_x))
        self.label_rot_y.setText("Rotation y: " + str(DroneController.dronecontrol.GetState().ori_y))
        self.label_rot_z.setText("Rotation z: " + str(DroneController.dronecontrol.GetState().ori_z))
        
        self.label_speed.setText("Linear Speed: " + str(DroneController.dronecontrol.GetState().linear_speed))
        self.label_angle_speed.setText("Angle Speed: " + str(DroneController.dronecontrol.GetState().a_v_x))

    # 移动功能
    def forward(self):
        DroneController.dronecontrol.DroneMoveByTime(veh.acc, 0, 0)

    def backward(self):
        DroneController.dronecontrol.DroneMoveByTime(-veh.acc, 0, 0)

    def left(self):
        DroneController.dronecontrol.DroneMoveByTime(0, -veh.acc, 0)

    def right(self):
        DroneController.dronecontrol.DroneMoveByTime(0, veh.acc, 0)

    def stop(self):
        pass

    def up(self):
        DroneController.dronecontrol.DroneMoveByTime(0, 0, -veh.acc)

    def down(self):
        DroneController.dronecontrol.DroneMoveByTime(0, 0, veh.acc)
        
    def change_weather(self, index):
        if index >= 0:
            weather = self.weather_selection.currentText()
            if weather == "Snow":
                DroneController.dronecontrol.SnowTeleport()
            elif weather == "Rain":
                DroneController.dronecontrol.RainTeleport()
            elif weather == "Fog":
                DroneController.dronecontrol.FogTeleport()
            elif weather == "Overcast":
                DroneController.dronecontrol.OvercastTeleport()
            elif weather == "Clear":
                DroneController.dronecontrol.ClearTeleport()
            elif weather == "Sand":
                DroneController.dronecontrol.SandTeleport()

    def on_algorithm_changed(self, index):
        """Algorithm下拉框选择变化时的处理"""
        if index >= 0:
            selected_algorithm = self.algorithm_combo.currentText()
            print(f"Selected Algorithm: {selected_algorithm}")


    def on_formation_changed(self, index):
        """Formation下拉框选择变化时的处理"""
        if index >= 0:
            selected_formation = self.formation_combo.currentText()
            print(f"Selected Formation: {selected_formation}")


    def on_task_changed(self, index):
        """Task下拉框选择变化时的处理"""
        if index >= 0:
            selected_task = self.task_combo.currentText()
            print(f"Selected Task: {selected_task}")


    def on_start_changed(self, index):
        if index >= 0:
            pos = self.start_combo.currentText()
            if pos == "9.8, -279, -22":
                DroneController.dronecontrol.SnowTeleport()
            elif pos == "213, -429, 19":
                DroneController.dronecontrol.OvercastTeleport()
            elif pos == "-315, -444, 12":
                DroneController.dronecontrol.RainTeleport()
            elif pos == "-335, 282, 19":
                DroneController.dronecontrol.SandTeleport()
            elif pos == "-152, 236, 66":
                DroneController.dronecontrol.ClearTeleport()
            elif pos == "-430, -91, 13":
                DroneController.dronecontrol.FogTeleport()
            else:
                print(self.start_combo.currentIndex())

    def EnableKeyboard(self):
        keyboard.hook(DroneController.dronecontrol.KeyboardControl)
        keyboard.wait()

window = AirDroneClientWindow()
window.initUI()
window.setInfo()
sys.exit(app.exec())