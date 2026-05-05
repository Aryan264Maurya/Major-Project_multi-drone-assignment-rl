# 🚁 Multi-Drone Task Assignment using DQN and PPO in AirSim

## 📌 Overview

This project focuses on solving the **multi-drone task assignment problem** using **Deep Reinforcement Learning (DRL)** and compares the performance with the **Hungarian Algorithm (optimal baseline)**.

The objective is simple:
👉 Assign each drone to exactly one task such that the **total travel distance is minimized**.

### 🔍 Implemented Approaches

* 🧠 **DQN (Deep Q-Network)**
* ⚙️ **PPO (Proximal Policy Optimization)**
* 📊 **Hungarian Algorithm (Baseline for optimal comparison)**

---

## ✨ Features

* Multi-drone task assignment system
* DQN-based assignment learning
* PPO-based assignment learning
* Hungarian optimal baseline comparison
* AirSim integration for real simulation
* Completion time measurement
* Flight trajectory visualization
* Training convergence plots:

  * Reward
  * Cost
  * Gap vs Hungarian
  * Loss
* Pre-trained models saved in `.pt` format

---

## 🧩 Problem Statement

Given:

* `n` drones
* `n` tasks

Assign each drone to a unique task such that:

[
\text{Total Euclidean Distance is minimized}
]

---

## ⚙️ Methodology

### 🧠 State Representation

* Drone positions
* Task positions
* Drone-task distance matrix

---

### 🎯 Action Space

* Each action represents a **valid permutation** of assigning drones to tasks

---

### 🏆 Reward Function

* Reward = **Negative total assignment cost**
* Lower distance ⇒ Higher reward

---

### 📊 Baseline

* The **Hungarian Algorithm** is used to compute the **optimal assignment**
* RL models are evaluated based on how close they get to this optimal solution

---

## 📁 Project Structure

```
approaches/
├── assignment_layer.py
├── baseline_hungarian.py
├── dqn_assignment.py
├── ppo_assignment.py
├── assignment_dqn.pt
├── ppo_assignment.pt
├── outputs/
│   ├── dqn/
│   │   ├── models/
│   │   └── plots/
│   └── ppo/
│       ├── models/
│       └── plots/
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install numpy scipy torch matplotlib airsim
```

### (Optional) Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install numpy scipy torch matplotlib airsim
```

---

## 🎮 AirSim Setup

Before running simulations:

1. Install and launch **AirSim**
2. Ensure the simulator is running properly
3. Verify drone names match the code:

```
Drone1
Drone2
Drone3
```

---

## 🏋️ Training

### Train DQN

```bash
python dqn_assignment.py --mode train --episodes 10000
```

### Train PPO

```bash
python ppo_assignment.py --mode train --episodes 10000
```

---

## 🚀 Run in AirSim

### Run DQN

```bash
python dqn_assignment.py --mode run --model_path assignment_dqn.pt --tests 5
```

### Run PPO

```bash
python ppo_assignment.py --mode run --model_path ppo_assignment.pt --tests 5
```

---

## 📊 Outputs Generated

### During Training

Saved automatically:

* Reward convergence
* Cost convergence
* Gap vs Hungarian
* Loss curve

📁 Location:

```
outputs/dqn/plots/
outputs/ppo/plots/
```

---

### During AirSim Testing

* Flight trajectory plots
* Completion time (console)
* Assignment cost comparison vs Hungarian

---

## 📈 Metrics Reported

For each test:

* Drone positions
* Task positions
* Selected assignment permutation
* RL assignment cost
* Hungarian optimal cost
* Cost difference (gap)
* Completion time
* Status (success / timeout)

---

## 🧪 Example Output

* Assignment chosen by RL model
* Completion time
* Saved trajectory plot path
* Cost comparison with optimal baseline

---

## ⚠️ Notes

* Training does **not** require AirSim
* AirSim is required only for `--mode run`
* Ensure correct `.pt` model path
* If drones get stuck:

  * Check AirSim connection
  * Reduce waiting time in code

---

## 🔮 Future Improvements

* Scale to larger number of drones/tasks
* Introduce obstacles in environment
* Handle dynamic tasks
* Multi-step RL environment
* Live flight visualization
* Runtime comparison (DQN vs PPO vs Hungarian)

---

## 👨‍💻 Author

**Aryan Maurya**
