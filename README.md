🚁 Multi-Drone Task Assignment using DQN, PPO, and Hungarian Optimization in AirSim
📌 Overview
This project solves the multi-drone task assignment problem using Deep Reinforcement Learning (DRL) and compares the results with the Hungarian Algorithm as the optimal baseline.
The system is extended to support:
Dynamic task reassignment when a UAV crashes or becomes unavailable
Multiple task-load scenarios: `n`, `2n`, and `3n` tasks
Multiple assignment features beyond distance, including battery, priority, speed, altitude, and direction alignment
AirSim-based simulation for real-time testing and trajectory visualization
The goal is to assign drones to tasks so that the overall mission cost is minimized while handling failures and comparing performance across different swarm sizes.
✨ Implemented Approaches
🧠 DQN (Deep Q-Network)
⚙️ PPO (Proximal Policy Optimization)
📊 Hungarian Algorithm as the optimal assignment baseline
🔁 Dynamic reassignment when a drone crashes or is removed from the active swarm
🧩 Problem Statement
Given:
`n` drones
`n`, `2n`, or `3n` tasks
Assign drones to tasks such that the total assignment cost is minimized.
The cost is no longer based on only Euclidean distance. It now also considers:
altitude difference
drone battery level
task priority
drone speed
alignment between drone motion and target direction
🛠️ Key Features
Multi-drone task assignment system
DQN-based assignment learning
PPO-based assignment learning
Hungarian optimal baseline comparison
AirSim integration for simulation and testing
Dynamic task reassignment after crash/failure
Support for `n`, `2n`, and `3n` task cases
Flight trajectory visualization
Completion time measurement
Training convergence plots:
Reward
Cost
Gap vs Hungarian
Loss
Saved model checkpoints in `.pt` format
⚙️ Methodology
1) State Representation
The state used by the RL agents includes:
Drone positions
Task positions
Drone battery levels
Drone speed values
Task priorities
Task completion mask
Distance information between drones and tasks
Current drone indicator
2) Action Space
Each action represents selecting a task for the current drone in the sequential assignment process.
3) Reward Function
Reward is defined as the negative assignment cost:
Lower cost → higher reward
Higher cost → lower reward
4) Optimal Baseline
The Hungarian Algorithm is used to compute the minimum-cost assignment and evaluate the quality of DQN and PPO decisions.
5) Dynamic Reassignment
If a UAV collides or becomes unavailable during execution:
the drone is removed from the active set
remaining tasks are reassigned to the surviving UAVs
the assignment is recomputed optimally for the active drones
This makes the system more realistic for UAV swarm missions where failures can happen unexpectedly.
📁 Project Structure
```bash
approaches/
├── assignment_layer.py
├── assignment_env.py
├── dqn_assignment.py
├── ppo_assignment.py
├── dynamic_reassignment_runner.py
├── outputs/
│   ├── dqn/
│   │   ├── models/
│   │   └── plots/
│   └── ppo/
│       ├── models/
│       └── plots/
```
📦 Requirements
Install dependencies:
```bash
pip install numpy scipy torch matplotlib airsim
```
Optional: Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install numpy scipy torch matplotlib airsim
```
🎮 AirSim Setup
Before running simulations:
Install and launch AirSim
Make sure the simulator is running correctly
Verify the drone names used in the code:
```text
Drone1
Drone2
Drone3
```
🏋️ Training
Train DQN
```bash
python dqn_assignment.py --mode train --task_multiplier 1
python dqn_assignment.py --mode train --task_multiplier 2
python dqn_assignment.py --mode train --task_multiplier 3
```
Train PPO
```bash
python ppo_assignment.py --mode train --task_multiplier 1
python ppo_assignment.py --mode train --task_multiplier 2
python ppo_assignment.py --mode train --task_multiplier 3
```
Each task multiplier should be trained separately so the saved model matches the state/action dimensions of that scenario.
🚀 Running in AirSim
Run DQN
```bash
python dqn_assignment.py --mode run --task_multiplier 1 --model_path outputs/dqn/models/dqn_t1.pt
python dqn_assignment.py --mode run --task_multiplier 2 --model_path outputs/dqn/models/dqn_t2.pt
python dqn_assignment.py --mode run --task_multiplier 3 --model_path outputs/dqn/models/dqn_t3.pt
```
Run PPO
```bash
python ppo_assignment.py --mode run --task_multiplier 1 --model_path outputs/ppo/models/ppo_t1.pt
python ppo_assignment.py --mode run --task_multiplier 2 --model_path outputs/ppo/models/ppo_t2.pt
python ppo_assignment.py --mode run --task_multiplier 3 --model_path outputs/ppo/models/ppo_t3.pt
```
🔁 Dynamic Task Reassignment
The project now supports failure-aware execution.
When a drone crashes or is removed:
its current assignment is dropped
remaining alive drones are reassigned to pending tasks
the Hungarian-based assignment is recomputed using the updated active swarm
This makes the system more realistic for UAV swarm missions where failures can happen unexpectedly.
📊 Features Used in Assignment Cost
The assignment layer now considers more than distance:
Euclidean distance
Altitude difference
Battery factor
Task priority
Speed factor
Direction alignment / turn penalty
This makes the comparison more meaningful for real UAV mission planning.
📈 Outputs Generated
During Training
Saved automatically in:
```text
outputs/dqn/plots/
outputs/ppo/plots/
```
Generated plots:
Reward convergence
Cost convergence
Gap vs Hungarian
Loss curve
During AirSim Testing
Flight trajectory plots
Completion time
RL vs Hungarian cost comparison
Crash recovery / reassignment logs
📋 Metrics Reported
For each test, the program reports:
Drone positions
Task positions
Selected assignment order
RL assignment cost
Hungarian optimal cost
Cost gap
Completion time
Success or timeout status
Recovery behavior during crashes
🧪 Experiment Cases
The system is evaluated in three task-load settings:
`n` drones : `n` tasks
`n` drones : `2n` tasks
`n` drones : `3n` tasks
This helps compare how DQN, PPO, and Hungarian behave under increasing workload.
🔮 Future Improvements
Larger UAV swarms
More realistic obstacle avoidance
Dynamic moving targets
Communication delay and packet loss
Multi-agent RL for decentralized control
Energy-aware mission planning
Real-time swarm formation control
👨‍💻 Author
Aryan Maurya