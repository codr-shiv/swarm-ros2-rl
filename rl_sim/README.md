# rl_sim: PPO Frontier Selection

A fast 2D simulator of the two-robot exploration task, used to train a PPO policy that decides **which frontier each robot explores next**. The trained policy then runs unchanged on the full Gazebo / SLAM / Nav2 stack through `ros_policy_node.py`.

| | rl_sim | Gazebo stack |
|---|---|---|
| Arena | parsed from the Gazebo world file and rasterised at 0.05 m | the same world file |
| Map | shared belief grid (unknown / free / occupied) | merged SLAM map |
| Sensor | 360-ray lidar, 3.5 m | LDS-01, 3.5 m |
| Frontiers | exact detection of `frontier_coordinator.py` (7×7 inflation, opening, > 15 px, 1.5 m dedup) | `frontier_coordinator.py` |
| Motion | 8-connected A* on the map inflated by 0.2 m, 0.15 m/s | Nav2 (Theta*, DWB, 0.15 m/s) |
| Speed | ~0.1 s per episode | ~5 min per episode |

## The learning problem
- **Step = one decision.** When a robot needs a goal, the policy picks one of the **12 nearest reachable frontiers** (`Discrete(12)`). Empty slots, frontiers closer than 0.6 m, the other robot's goal, and already-explored spots are **masked** (MaskablePPO). The robot then drives there while both lidars map, until the next decision.
- **Observation (153 numbers):**
  - **12 features per candidate:** valid flag; position; offset from the robot; distance; distance to the other robot and to its goal; frontier size; unknown fraction within 1 m; heading alignment; the heuristic's pick flag.
  - **9 global features:** both robots' positions, the other robot's goal, the explored fraction, the elapsed time.
- **Reward:** +0.1 per newly known m², −0.01 per simulated second. The episode ends when nothing is left to explore, so **finishing faster scores higher**.
- **Algorithm:** MaskablePPO with an MLP of 2×128 for each of the policy and value heads. Settings:

  | Setting | Value |
  |---|---|
  | Parallel envs | 8 |
  | Steps per env per update | 256 |
  | Batch size | 256 |
  | Epochs per update | 10 |
  | Learning rate | 3e-4, decaying linearly |
  | γ / λ | 0.99 / 0.95 |
  | Clip range | 0.2 |
  | Entropy coefficient | 0.01 |

- **Why it trains reliably:**
  - the actions are stable and masked
  - one action is a whole trip (episodes are ~10 decisions)
  - the features are compact
  - the reward is small and well scaled
  - the simulator is deterministic

  The policy reaches heuristic level within 10k decisions and settles by ~20k (`graphs/01_training_return.svg`).

## Commands
Run inside the ROS 2 environment (e.g. `distrobox enter ros-humble`), from the repository root.

```bash
cd ~/swarm

# one-time Python setup (CPU torch is enough)
pip3 install --user torch --index-url https://download.pytorch.org/whl/cpu
pip3 install --user -r rl_sim/requirements.txt
pip3 uninstall -y setuptools        # torch installs a setuptools that breaks colcon on Humble
colcon build --symlink-install

# 2D simulator: compare the shipped policy with the baselines
python3 rl_sim/evaluate.py --model models/ppo_frontier_policy.zip

# train a new policy (~25 min); progress is printed every 10k decisions
python3 rl_sim/train.py

# run a policy on the full Gazebo stack, headless, and log coverage (~5 min per run)
rl_sim/gazebo_test.sh ppo 240 models/ppo_frontier_policy.zip
rl_sim/gazebo_test.sh heuristic 240

# statistics and charts from everything in results/
python3 rl_sim/analyze_gazebo_tests.py
python3 rl_sim/make_graphs.py
```

**Training output:**
```
[  20008 decisions] policy: return 1.827, explored in 65.5 s (7 decisions, explored) | heuristic 1.751 / 72.5 s  <- best, saved
```
The run directory `~/rl_sim_runs/ppo_<time>/` gets:
- `best_model.zip` and `final_model.zip`
- `eval_history.csv`, for the training-curve graphs (copy it to `results/training/` to update them)
- `tb/`, for `tensorboard --logdir ~/rl_sim_runs`

| `train.py` flag | Default | Meaning |
|---|---|---|
| `--timesteps` | 300000 | total decisions |
| `--num-envs` | 8 | parallel simulators (CPU processes) |
| `--eval-every` | 10000 | decisions between evaluations / best-model saves |
| `--seed` | 0 | PPO random seed |
| `--run-dir` | `~/rl_sim_runs/ppo_<time>` | output folder |

`gazebo_test.sh <ppo|heuristic|nearest> [sim_seconds=240] [model.zip]` starts Gazebo (no window), SLAM, map merge and Nav2 on its own ROS domain. It runs the policy node, writes `results/gazebo_runs/<policy>_<time>.csv` (known area every second), keeps launch logs in `~/rl_sim_runs/gazebo_tests/`, and shuts everything down.

## Using the policy in the normal pipeline
Start terminals 1–4 as usual on the benchmark arena (`GAZEBO_WORLD_SEED=42` in terminal 1), then use this instead of `frontier_exploration.launch.py`:
```bash
python3 rl_sim/ros_policy_node.py --model models/ppo_frontier_policy.zip
```
**How it works:**
- The node keeps a 2D-simulator instance of the arena.
- Every second, it fills that instance with the real state: the merged `/map` resampled onto the 0.05 m grid, robot poses and headings from TF, the current goals, the elapsed sim time, and recently failed goals.
- It asks the simulator's own decision code for the observation and mask, so the policy input is built exactly as in training.
- The chosen frontier goes to that robot's Nav2.
- **A goal ends when:** Nav2 reports success or failure, the area around it is fully mapped, or 90 s pass. Failed goals are skipped for 30 s.
- `--policy heuristic|nearest` runs the same node with a hand-designed rule instead, for comparisons with identical goal handling.

## Files
| File | Role |
|---|---|
| `core/world.py` | builds the ground-truth grid from the Gazebo world generator |
| `core/lidar.py` | vectorised 360° raycaster |
| `core/planner.py` | A* on the inflated map |
| `core/frontiers.py` | frontier detection identical to `frontier_coordinator.py` |
| `envs/frontier_env.py` | the Gymnasium environment (+ heuristic and nearest baselines) |
| `train.py` | MaskablePPO training with deterministic evaluation and best-model saving |
| `evaluate.py` | PPO vs heuristic / nearest / random in the 2D simulator, with final-map images |
| `ros_policy_node.py` | runs a policy (or a baseline) on the Gazebo stack |
| `gazebo_test.sh` | one headless Gazebo run with coverage logging |
| `analyze_gazebo_tests.py` | paired statistics over all Gazebo runs |
| `make_graphs.py` | every chart in `graphs/` from the data in `results/` |

## Results
On the benchmark arena, over 8 alternating paired Gazebo runs:
- **Area:** PPO mapped **41.8 ± 2.5 m²** after 240 s, against **36.2 ± 4.0 m²** for the heuristic. That's +15 %, better in 8/8 pairs (exact paired permutation p = 0.008).
- **Speed:** it reached 30 m² about 30 % sooner.

Full analysis: [../heuristic-vs-rl.md](../heuristic-vs-rl.md). Charts: [../graphs/](../graphs/).

**Sim-to-sim gap:** Gazebo maps 3–4× slower in absolute terms, because SLAM builds the map gradually and Nav2 adds rotations and recoveries. The relative advantage still carries over.
