# Demo Videos

Each video is a top-down view of the benchmark arena in the 2D simulator (`rl_sim`).

**How to read the frames:**
- **Colours:** grey = unknown, white = mapped free space, black = obstacles.
- **Robots:** orange and blue dots are `robot1` and `robot2`; a cross of the same colour is that robot's current goal frontier.
- **Overlay:** the text at the top shows simulated time and mapped area.
- **Speed:** the simulation advances 0.5 s per frame and plays at 10 frames per second, i.e. 5× real time.

To regenerate all of these:
```bash
cd ~/swarm/final-product
python3 rl_sim/evaluate.py --model models/ppo_frontier_policy.zip --video --out-dir media/demo-videos
```
(The side-by-side and GIF are made with `ffmpeg` from the two single videos.)

| Video | What it demonstrates | Outcome |
|---|---|---|
| [`sim2d_ppo_vs_heuristic.mp4`](sim2d_ppo_vs_heuristic.mp4) ([GIF](sim2d_ppo_vs_heuristic.gif)) | **The main result.** Left: the trained PPO policy. Right: the hand-designed heuristic of `frontier_coordinator.py`, in the same arena from the same start. | PPO maps faster: at t = 20.5 s it knows 48.4 m², against 43.3 m² for the heuristic. It finishes in 65.5 s vs 72.5 s. |
| [`sim2d_ppo.mp4`](sim2d_ppo.mp4) (13 s) | **The RL policy alone.** Each robot picks one of the 12 nearest reachable frontiers; the robots split the arena without explicit partitioning rules. | Full exploration in 65.5 s with 7 decisions. |
| [`sim2d_heuristic.mp4`](sim2d_heuristic.mp4) (15 s) | **The heuristic brain.** It uses distance plus penalties for crowding the other robot's goal or entering its half. | Full exploration in 72.5 s with 8 decisions. |
| [`sim2d_nearest.mp4`](sim2d_nearest.mp4) (21 s) | **Baseline: nearest frontier.** Each robot always takes the closest frontier, with no coordination. | Full exploration in 105.5 s. |
| [`sim2d_random.mp4`](sim2d_random.mp4) (29 s) | **Baseline: random valid frontier.** A lower bound on performance. | Full exploration in 143.0 s. |

## The full ROS 2 stack in Gazebo
The Gazebo evaluation (8 paired headless runs) is logged as coverage data, not video: see `results/gazebo_runs/` and `graphs/04_gazebo_coverage_over_time.svg`.

To record the full stack visually:
1. Start the stack with the Gazebo window (`gui:=true`, the default) and RViz using `config/multi_robot_exploration_cinematic.rviz`.
2. Screen-record both windows while running the heuristic brain (terminal 5: `frontier_exploration.launch.py`) and the RL brain (`rl_sim/ros_policy_node.py`). For the RL brain, launch terminal 1 with `GAZEBO_WORLD_SEED=42`.
