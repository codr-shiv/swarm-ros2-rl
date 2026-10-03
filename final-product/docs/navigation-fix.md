# Known Issue (Fixed): Robots Getting Stuck Near Obstacles

## Symptom
Robots sometimes drove up to an obstacle and stopped for the rest of the run. The Nav2 logs showed the same messages repeating:

| Message | Meaning |
|---|---|
| `Either of the start or goal pose are an obstacle!` (planner) | The planner thinks the robot's own position is inside an obstacle |
| `No valid trajectories` (controller) | Every candidate motion collides |
| `Collision Ahead` (behavior server) | The back-up recovery is blocked |

It happened in **both** exploration modes, so it was a navigation problem, not something the RL policy learned.

## Diagnosis
The investigation found two causes.

### 1. The main cause: "ghost" obstacles in the global costmap
- **How the ghost forms:** at start-up each robot's lidar sees the *other robot*, and SLAM marks it as an obstacle. That blob went into the merged `/map`, and Nav2's global costmap loaded `/map` as a **static layer**. The ghost stays there after the other robot has driven away.
- **How a robot gets trapped:** the space next to a ghost looks like a frontier, so a robot gets sent there. Once its centre is on or next to the ghost, its own lidar can never clear it, because the LDS-01 cannot see closer than ~0.12 m. From then on, every plan fails with "start pose is an obstacle", and the robot is stuck until the run ends.
- **Evidence:** in a stuck run, robot 1's first goal was (3.03, 0.23), almost exactly robot 2's spawn point. Every later goal for robot 1 failed instantly with "start … is an obstacle" (130 times), with blocked back-ups in between.

### 2. A contributing cause: a very small safety margin
`inflation_radius: 0.15` m around obstacles was only ~4.5 cm more than the robot radius (0.105 m). With a steep `cost_scaling_factor: 8.0` and a weak DWB obstacle critic (`ObstacleFootprint.scale: 0.1`), planned paths and the controller hugged walls. Robot 2's local costmap also used a different value (0.25) from robot 1's (0.15).

### Not the cause: the 2D training environment
It already accounts for the robot's size: paths and goals use a map inflated by the robot radius plus a margin. Its margin (0.2 m) was, however, chosen for the old Nav2 settings, so it was raised to stay consistent (see below).

## Fix

| File | Change | Why |
|---|---|---|
| `config/nav2_params_robot{1,2}.yaml`, global costmap | **Removed the static layer.** Obstacles come from live lidar (obstacle layer with raytrace clearing), in a fixed 10 × 10 m window | Ghosts of the other robot are no longer permanent. Free space seen again is cleared. Exploration decisions still use the merged `/map` |
| same, both costmaps | `inflation_radius` 0.15 → **0.35** (robot 2: 0.25 → 0.35), `cost_scaling_factor` 8.0 → **3.5** | Plans keep a real margin from walls. Both robots are now configured identically |
| same, DWB controller | `ObstacleFootprint.scale` 0.1 → **0.5** | The controller stops cutting corners against obstacles |
| same | `footprint_padding: 0.02` | A small extra margin around the robot circle |
| `rl_sim/core/world.py` | `ROBOT_INFLATION` 0.20 → **0.30** m | The training simulator's clearance for paths and goals matches the new Nav2 margin. **The policy was retrained** with it |

## Result
Averages over 8 headless Gazebo runs of 240 s per policy, before vs after the fix. Raw counts are in `results/nav2_stuck_counts_before_fix.csv` and `results/nav2_stuck_counts_after_fix.csv`, produced by `python3 rl_sim/count_nav2_failures.py`.

| Per run | PPO before | PPO after | Heuristic before | Heuristic after |
|---|---|---|---|---|
| "start or goal pose are an obstacle" | 29.5 | **1.5** | 54.8 | **0.4** |
| back-up blocked ("Collision Ahead") | 2.9 | 1.4 | 4.2 | 2.1 |
| failed goals | 1.4 | **0.0** | 2.4 | **0.4** |
| known area at 240 s (m²) | 41.8 | 41.3 | 36.2 | **38.4** |

The robots no longer get pinned. "No valid trajectories" still appears from time to time, mostly when the two robots pass close to each other. These are brief: the controller recovers, and goals don't fail.

**Effect on the RL comparison:** the heuristic gained the most from the fix, because it got stuck most often. PPO still maps faster (30 m² reached 21 % sooner, better in 8/8 pairs, p = 0.008), but the gap in final area at 240 s is smaller than before the fix. The current numbers are in [heuristic-vs-rl.md](heuristic-vs-rl.md). The pre-fix data is kept in `results/gazebo_runs_before_nav2_fix/` for reference.
