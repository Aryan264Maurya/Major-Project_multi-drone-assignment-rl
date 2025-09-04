---
license: apache-2.0
tags:
- Swarm
- UAVS
- Autonoumous
- Flying
- Simulator
pretty_name: U2USim-2
size_categories:
- 10K<n<100K
---

## Abstract
Swarm UAV autonomous flight for Long-Horizon (LH) tasks is crucial for advancing the low-altitude economy. 
However, existing methods focus only on specific basic tasks due to dataset limitations, failing in real-world deployment for LH tasks. 
LH tasks are not mere concatenations of basic tasks, requiring handling long-term dependencies, maintaining persistent states, and adapting to dynamic goal shifts. 
This paper presents U2UData-2, the first large-scale swarm UAV autonomous flight dataset for LH tasks and the first scalable swarm UAV data online collection and algorithm closed-loop verification platform. 
The dataset is captured by 15 UAVs in autonomous collaborative flights for LH tasks, comprising 12 scenes, 720 traces, 120 hours, 600 seconds per trajectory, 4.32M LiDAR frames, and 12.96M RGB frames. 
This dataset also includes brightness, temperature, humidity, smoke, and airflow values covering all flight routes. 
The platform supports the customization of simulators, UAVs, sensors, flight algorithms, formation modes, and LH tasks. 
Through a visual control window, this platform allows users to collect customized datasets through one-click deployment online and to verify algorithms by closed-loop simulation. 
U2UData-2 also introduces an LH task for **wildlife conservation** and provides comprehensive benchmarks with 9 SOTA models. 
U2UData-2 can be found at https://fengtt42.github.io/U2UData-2/.

## Helper
If you need the full dataset, you can send an email to obtain the Baidu Cloud link for downloading.
PPM is a binary image format. If you need PNG format, we provide conversion code, which you can download and convert by yourself.
I have full parameter information for each trajectory. If you need other parameter information, such as IMU, detailed drone parameters, etc., you can contact me by email at any time.

## Cite
@inproceedings{feng2025u2udata2scalableswarmuavs,

    title={U2UData-2: A Scalable Swarm UAVs Autonomous Flight Dataset for Long-horizon Tasks}, 

    author={Tongtong Feng and Xin Wang and Feilin Han and Leping Zhang and Wenwu Zhu},

    year={2025},

    url={https://arxiv.org/abs/2509.00055}

}

U2UData-2: https://huggingface.co/papers/2509.00055

## Web Link
https://fengtt42.github.io/U2UData-2/



## 1. 文件分类
- U2USim_2_Windows_version 仿真环境的windows打包版本
- U2USim_2_Linux_version 仿真环境的Linux20.04打包版本
- Settings.json 配置文件　放在\home\用户名\Documents\AirSim或者C:\Users\用户名\Documents\AirSim下 是对插件的一些预定义设置
- AirDrone 基于API的算法控制、键盘控制、可视化交互控制器

## 2. 系统需求
### 2.1 PC需求
- 内存至少32G以上
- 显卡 RTX 4080，7900 XTX
- CPU 主流CPU应该都行
### 2.2 手柄 (经过测试的)
- Xbox One， Xbox series X/S 的手柄都行，但是Xbox精英2代手柄不行
- PS5 DuelSense controller 

## 3. 运行环境需求和控制方式
### 3.1 U2USim_2_Windows_version
- 不需要安装额外的运行库
- 手柄控制 左手摇杆左右控制转向，上下飞机高度，右手控制飞机倾斜（移动），但是会让飞机下降
- 仿真环境的键盘控制（在使用了EnableKeyboardControl之后无效）
    - Backspace 飞机位置重置
    - 键盘1234 打开图像可视化窗口
    - F Fpv（第一人称）模式
    - / 飞机固定在视角中下部
    - End 重置整个仿真环境（重新打开关卡）
    - R 录制传感器数据，鼠标点右下角红色按钮是同样的功能

### 3.2 U2USim_2_Linux_version
- 控制方式和Windows相同
- ROS信息获取参考 https://zhuanlan.zhihu.com/p/678127741
- Airsim ROS package 仓库 https://github.com/seventyzlp/Codename-Aleph

### 3.3 AirDrone
- 安装python运行库
    - PyQt5
    - keyboard
    - numpy
    - msgpack-rpc-python
- 需要运行的主文件是AirDroneClient.py
- 需要先打开仿真环境后，再启动这个控制器，不然会黑屏
- 在点击EnableKeyboardControl按钮后，会强制接管所有键盘输入事件，关闭需要结束程序
    - 键盘控制K是起飞，WASD 前后左右移动，上下 飞机高度升高降低， 左右控制飞机转向
- 点击按键控制飞机不受影响
- 在使用API控制飞机后，手柄的控制会被自动禁用，按下ExitApiControl按钮后可以重新使用手柄


