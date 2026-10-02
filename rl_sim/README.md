# rl_sim: PPO frontier selection on a fixed world

A fast 2D simulator of the two-robot exploration task, used to train a PPO policy that picks **which frontier each robot explores next**. It trains on **one fixed world**: the same world Gazebo builds for `GAZEBO_WORLD_SEED=<seed>`. The policy is meant to be used on that world; it isn't expected to generalize to others.

Training takes minutes on a laptop CPU (about 0.1 s per episode, vs about 5 min per episode in Gazebo).

## What is simulated
| | rl_sim | Gazebo stack |
|---|---|---|
| World | parsed from the SDF that `generate_random_world(seed)` writes, rasterised at 0.05 m | same SDF |
| Map | shared belief grid (−1 unknown / 0 free / 100 occupied) | merged SLAM map |
| Sensor | 360-ray lidar, 3.5 m | LDS-01, 3.5 m |
| Frontiers | coordinator's exact detection (7×7 inflation, opening, > 15 px, 1.5 m dedup) | `frontier_coordinator.py` |
| Motion | 8-connected A* on the map inflated by 0.2 m, 0.15 m/s | Nav2 (Theta*, DWB, 0.15 m/s) |
| Goal done | path finished, or no unknown left within 1 m of the goal | reached / vanished |

**Not simulated:** Nav2 failures, recovery behaviours, SLAM drift, and robot-robot blocking. Navigation is ideal.

## The RL problem
- **Step = one decision.** When a robot needs a goal, the policy picks one of the **12 nearest reachable frontiers** (`Discrete(12)`). Empty slots, frontiers closer than 0.6 m, the other robot's goal, and already-explored spots are **masked** (MaskablePPO). The robot then drives there while both lidars map, until the next decision.
- **Observation (153 floats):**
  - **12 features per candidate:** valid flag; position; offset from the robot; distance; distance to the other robot and to its goal; frontier size; unknown fraction within 1 m; heading alignment; frontier_coordinator's pick flag.
  - **9 global features:** both robots' positions, the other robot's goal, the explored fraction, the elapsed time.
- **Reward:** +0.1 per newly known m², −0.01 per simulated second. The episode ends when nothing is left to explore (or at 600 s), so **finishing faster scores higher**.
- **Why it converges:**
  - the actions are stable and masked
  - one action is a whole trip (episodes are ~10 decisions)
  - the features are compact
  - the reward is small and well scaled
  - the simulator is deterministic, so there's no noise

## Run it (inside the `ros-humble` distrobox, from the repo root)

```bash
cd ~/swarm
# once: CPU torch + packages (skip if already installed)
pip3 install --user torch --index-url https://download.pytorch.org/whl/cpu
pip3 install --user -r rl_sim/requirements.txt
pip3 uninstall -y setuptools        # torch's setuptools breaks colcon on Humble

# baselines on the world you'll use (instant)
python3 rl_sim/evaluate.py --world-seed 42

# train (~25 min for the default 300k decisions with 8 envs; converges well before that)
python3 rl_sim/train.py --world-seed 42

# compare the trained policy with the baselines; saves final_map_*.png next to the model
python3 rl_sim/evaluate.py --model ~/rl_sim_runs/world42_<time>/best_model.zip --world-seed 42
```

During training, a line is printed every 10k decisions:
```
[  20008 decisions] policy: return 1.827, explored in 65.5 s (7 decisions, explored) | heuristic 1.751 / 72.5 s  <- best, saved
```
**Converged** means the policy's return stops changing and is at or above the heuristic's. On world 42 that happens by about 20k decisions. TensorBoard: `tensorboard --logdir ~/rl_sim_runs`.

| Flag | Default | Meaning |
|---|---|---|
| `--world-seed` | 42 | which world (same number as `GAZEBO_WORLD_SEED`) |
| `--timesteps` | 300000 | total decisions |
| `--num-envs` | 8 | parallel simulators (CPU processes) |
| `--eval-every` | 10000 | decisions between evaluations / best-model saves |
| `--seed` | 0 | PPO seed |

## Files
| File | Role |
|---|---|
| `core/world.py` | builds the ground-truth grid from `generate_random_world(seed)` |
| `core/lidar.py` | vectorised 360° raycaster |
| `core/planner.py` | A* on the inflated map |
| `core/frontiers.py` | frontier detection identical to `frontier_coordinator.py` |
| `envs/frontier_env.py` | the Gymnasium environment (+ heuristic and nearest baselines) |
| `train.py` | MaskablePPO training with deterministic evaluation and best-model saving |
| `evaluate.py` | PPO vs heuristic / nearest / random on the same world, with final-map PNGs |

## Using the policy in Gazebo (next step)
To run the trained policy on the real stack, launch Gazebo with the same seed (`GAZEBO_WORLD_SEED=42 ros2 launch multi_robot_exploration spawn_two_turtlebots.launch.py`). Terminal 5 then needs a node that builds the same 153-number observation from `/map`, TF and the robots' goals, and sends the chosen goal to Nav2. That node isn't written yet.
