# Multi-Robot Autonomous Frontier Exploration (ROS 2 + Reinforcement Learning)

Two TurtleBot3 robots cooperatively map an unknown arena: Gazebo simulation, per-robot SLAM, map merging, Nav2 navigation, and a frontier-selection brain that is either hand-designed or learned with PPO. The learned policy maps the benchmark arena **faster** than the heuristic: it reaches 30 m² **21 % sooner** (8/8 paired Gazebo runs, p = 0.008).

| Folder | Contents |
|---|---|
| [`final-product/`](final-product/) | **The deployable system:** ROS 2 package, RL code, trained model, results, graphs, architecture diagrams, demo videos, presentation. Start with [final-product/README.md](final-product/README.md). |
| [`resource-guide/`](resource-guide/) | Curated foundational material (ROS 2, SLAM, Nav2, frontier exploration, reinforcement learning) to learn before working on the code |

Quick start:
```bash
cd final-product && colcon build --symlink-install && source install/setup.bash
```
Then follow [final-product/README.md §3](final-product/README.md#3-how-to-run).
