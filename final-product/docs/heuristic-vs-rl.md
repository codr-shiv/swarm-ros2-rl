# Heuristic vs RL Frontier Selection

## Summary

A PPO frontier-selection policy, trained only in a lightweight 2D simulator of the benchmark arena, was transferred zero-shot to the full Gazebo/ROS 2 stack (two TurtleBot3s, slam_toolbox, map merging, Nav2). On the benchmark arena it **explored faster than the hand-designed heuristic of `frontier_coordinator.py`**:

| | PPO | Heuristic | Difference |
|---|---|---|---|
| Known area after 240 s (Gazebo, 8 paired runs) | **41.8 ± 2.5 m²** | 36.2 ± 4.0 m² | **+5.5 m² (+15 %)**, PPO ahead in **8/8** pairs, p = 0.008 |
| Time to map 30 m² (Gazebo) | **121 ± 26 s** | 173 ± 44 s | **−53 s (−30 %)**, PPO faster in 7/8, p = 0.031 |
| Time to fully explore (2D simulator) | **65.5 s** | 72.5 s | −7 s (−10 %), deterministic |

All charts are in [`../graphs/`](../graphs/); raw data is in [`results/`](../results/).

---

## 1. Setup

### 1.1 The decision being learned
When a robot needs a new goal, the coordinator picks one of the current frontiers (boundaries between mapped and unmapped space), and Nav2 drives the robot there. Only this choice differs between the two policies. Everything else is identical: frontier detection, goal handling, SLAM, map merging and Nav2.

- **Heuristic** (`frontier_coordinator.py`): picks the frontier with the lowest
  `cost = distance + 50 / dist(frontier, other robot's goal) + 50 · max(0, distance into the other robot's half)`.
- **PPO policy:** a 2×128 MLP that sees 12 candidate frontiers (nearest first) with 12 features each, plus 9 global features (153 numbers in total). It outputs which candidate to take; invalid slots are masked.

### 1.2 Training (rl_sim, 2D)
- **Arena:** parsed from the same world file Gazebo loads (`generate_random_world`), rasterised at 0.05 m.
- **Sensing and frontiers:** 360° lidar with 3.5 m range, and the coordinator's exact frontier detection.
- **Motion:** A* on a 0.2 m-inflated map at 0.15 m/s.
- **RL formulation:**

  | Item | Value |
  |---|---|
  | Step | one frontier decision for one robot |
  | Reward | +0.1 per newly mapped m², −0.01 per simulated second |
  | Episode end | when nothing is left to explore |
  | Algorithm | MaskablePPO (sb3-contrib) |
  | Parallel envs | 8 |
  | Training length | 300k decisions (~25 min CPU) |

- **Convergence:** evaluated every 10k decisions. The deterministic policy matched the heuristic at 10k decisions, beat it at 20k, and stayed **unchanged from 20k to 300k**, which indicates convergence. Model: [`models/ppo_frontier_policy.zip`](../models/).

![Training return](../graphs/01_training_return.svg)
![Training explore time](../graphs/02_training_explore_time.svg)

**PPO training curves** (TensorBoard log, also in [`results/training/`](../results/training/)):

![PPO training curves](../graphs/20_training_dashboard.svg)

They show clean convergence:
- **Reward:** the mean episode reward rises past the heuristic's within ~30k decisions and then stays flat.
- **Entropy:** policy entropy falls to ≈ 0, so the policy becomes confident.
- **Value function:** explained variance reaches ≈ 1.
- **Update size:** the KL divergence and clip fraction peak while the policy is changing, then drop to ≈ 0. Later updates no longer change it.

Individual charts: `../graphs/11_…` to `../graphs/19_…`.

### 1.3 Transfer to Gazebo (`rl_sim/ros_policy_node.py`)
Every second, the node rebuilds the training observation from the real system: the merged `/map` resampled onto the training grid, robot poses from TF, and the current Nav2 goals. It then runs the same decision code as in training. **Both policies run in this same node** (`--policy ppo` / `--policy heuristic`). The goal-ending rules are also identical for both: Nav2 success or failure, area around the goal fully mapped, or a 90 s timeout, with a 30 s blacklist for failed goals.

### 1.4 Gazebo protocol (`rl_sim/gazebo_test.sh`)
- **Each run:** a fresh headless stack: Gazebo on the benchmark arena, then SLAM, map merging and Nav2. The policy runs for **240 s of simulated time**, and the merged map's known area is logged every second.
- **Design:** 8 PPO runs and 8 heuristic runs, **alternated** (PPO, heuristic, PPO, …), all on one laptop (8 cores / 32 GB). The k-th PPO run is paired with the k-th heuristic run, so slow drift in machine state affects both sides of a pair equally.
- **Statistics:** mean ± sample std, mean paired difference, and an **exact two-sided paired permutation (sign-flip) test** over all 2⁸ = 256 sign assignments. It makes no normality assumption. With 8 pairs, the smallest achievable p is 2/256 ≈ 0.0078.

---

## 2. Results

### 2.1 2D simulator (deterministic)
| Policy | Return | Time to explore fully | Decisions |
|---|---|---|---|
| **PPO** | **1.827** | **65.5 s** | 7 |
| Heuristic | 1.751 | 72.5 s | 8 |
| Nearest frontier | 1.425 | 105.5 s | 9 |
| Random (valid) | 1.053 | 143.0 s | 9 |

![2D simulator comparison](../graphs/03_sim2d_time_to_explore.svg)

### 2.2 Gazebo (8 paired runs, 240 s each)
| Metric | PPO mean ± std | Heuristic mean ± std | Mean paired diff (PPO − H) | PPO better in | p |
|---|---|---|---|---|---|
| Known area at 60 s (m²) | 21.4 ± 1.5 | 19.5 ± 1.7 | +1.8 | 7/8 | 0.039 |
| Known area at 120 s (m²) | 30.6 ± 3.7 | 26.5 ± 4.5 | +4.1 | 7/8 | 0.125 |
| Known area at 180 s (m²) | 37.8 ± 2.1 | 31.4 ± 4.6 | +6.4 | 7/8 | 0.023 |
| **Known area at 240 s (m²)** | **41.8 ± 2.5** | 36.2 ± 4.0 | **+5.5** | **8/8** | **0.008** |
| Mean known area over 0–240 s (m²) | 28.9 ± 1.8 | 24.7 ± 2.8 | +4.2 | 7/8 | 0.031 |
| **Time to reach 30 m² (s)** | **120.5 ± 25.5** | 173.1 ± 44.0 | **−52.6** | 7/8 | 0.031 |
| Failed goals per run | 1.4 ± 0.7 | 2.4 ± 3.0 | −1.0 | 5/8 | 0.50 |
| Decisions per run | 21.0 ± 1.9 | 17.8 ± 1.3 | +3.3 | — | 0.008 |

![Known area over time](../graphs/04_gazebo_coverage_over_time.svg)
![Known area at checkpoints](../graphs/05_gazebo_area_at_checkpoints.svg)
![Paired difference at 240 s](../graphs/06_gazebo_paired_difference_240s.svg)
![Time to 30 m²](../graphs/07_gazebo_time_to_30m2.svg)
![Mean area over the run](../graphs/08_gazebo_mean_area.svg)
![Failed goals](../graphs/09_gazebo_failed_goals.svg)
![Decisions](../graphs/10_gazebo_decisions.svg)

**Per-run results** (known area in m² at 60 / 120 / 180 / 240 s, failed goals in brackets):

| Pair | PPO | Heuristic |
|---|---|---|
| 1 | 21.9 / 28.6 / 35.3 / **38.9** (1) | 19.1 / 27.1 / 30.6 / 32.2 (0) |
| 2 | 23.1 / 31.6 / 41.2 / **46.5** (2) | 19.2 / 25.3 / 30.0 / 35.0 (3) |
| 3 | 22.3 / 35.7 / 39.4 / **40.3** (2) | 22.0 / 32.0 / 37.5 / 39.9 (0) |
| 4 | 23.0 / 34.8 / 38.4 / **39.9** (2) | 19.3 / 21.0 / 27.9 / 33.1 (3) |
| 5 | 19.2 / 27.8 / 38.0 / **40.9** (0) | 17.1 / 23.2 / 26.5 / 37.0 (1) |
| 6 | 20.3 / 25.1 / 36.5 / **43.1** (1) | 22.0 / 34.1 / 38.9 / 42.3 (0) |
| 7 | 21.2 / 32.6 / 38.7 / **43.8** (1) | 18.3 / 23.9 / 32.6 / 39.0 (3) |
| 8 | 19.8 / 28.8 / 35.0 / **40.7** (2) | 19.3 / 25.3 / 27.2 / 31.4 (9) |

---

## 3. Analysis

### 3.1 What the numbers show
- **Consistent advantage at the end of the run.** PPO had more of the map known at 240 s in **all 8 pairs** (+0.4 to +11.5 m²). Under the null hypothesis of no difference, the chance of this is 2/256 (p = 0.008). The mean gain is **+5.5 m², about 15 %**.
- **Faster mapping, not just a better endpoint.** PPO reached 30 m² about **53 s sooner** (−30 %, p = 0.031), and its average known area over the whole run was 4.2 m² higher (p = 0.031). The coverage curve separates after the first minute and stays apart.
- **More reliable.** PPO's results are tighter: std 2.5 m² vs 4.0 m² at 240 s, and 26 s vs 44 s for time-to-30 m². The heuristic had a few slow runs (pairs 4 and 8). In those runs the robots spent a long time on low-yield goals, and pair 8 also had 9 failed goals.
- **Not significant at 120 s** (p = 0.125). Two pairs (3 and 6) were close or favoured the heuristic in the middle of the run, though PPO still ended ahead in both. The advantage builds over the run rather than being uniform at every moment.
- **Failed goals are not a differentiator.** The difference (1.4 vs 2.4) isn't significant and is driven by one outlier heuristic run.

### 3.2 Why PPO is faster (behaviour)
PPO makes **more, shorter trips**: 21 vs 18 decisions in the same 240 s. The heuristic's large penalties (50× for being near the other robot's goal, 50× per metre into the other robot's half) push it towards distant frontiers in its "own" half. The trained policy is willing to take nearer frontiers, and it redirects as soon as an area is mapped. In the 2D simulator it also finished with fewer decisions (7 vs 8), choosing frontiers whose surroundings reveal more at once. Both behaviours reduce driving time per new m² mapped.

### 3.3 Transfer from 2D to Gazebo
| | 2D simulator | Gazebo |
|---|---|---|
| Area known at start | ~30 m² (ideal 360° scan) | ~4 m² (SLAM map builds up gradually) |
| Time to map most of the arena | ~65–70 s | > 240 s |
| PPO vs heuristic | −10 % time to full exploration | +15 % area at 240 s, −30 % time to 30 m² |

Gazebo is 3–4× slower in absolute terms. slam_toolbox only integrates scans as the robots move, and Nav2 adds rotation and recovery time. The **relative advantage still transferred and was even larger.** A plausible reason: in Gazebo, the heuristic's long trips to its "own half" cost more real driving time.

---

## 4. Limitations and threats to validity

1. **Arena.** Training and evaluation use the benchmark arena. Performance on other layouts was not measured, so no claim is made about them.
2. **Sample size.** 8 pairs is modest. The 240 s result is robust (8/8, the minimum achievable p-value), but metrics with p ≈ 0.02–0.04 would not all survive a strict multiple-comparison correction (8 metrics; Bonferroni α ≈ 0.006). The primary, pre-chosen metric, **area at 240 s**, stays below 0.01.
3. **240 s horizon.** Neither policy fully explored the arena within 240 s in Gazebo, so "time to full exploration" wasn't measured there. Longer runs would show whether PPO also finishes first or the heuristic catches up.
4. **Same-node comparison.** The heuristic was run as a decision rule inside `ros_policy_node.py`, with the same goal handling as PPO. It was not run as the original `frontier_coordinator` node, whose goal handling differs (stuck detection, collision pausing, replanning every 3 s). The comparison isolates the decision rule, which is the claim being made, but isn't a comparison against the original node as a whole.
5. **One interrupted run.** The laptop went to sleep during the heuristic run of pair 6. Every process was frozen and later resumed. All timing is in simulated time, so the result is unaffected in principle, but that run's startup took longer in wall time.
6. **Simulation only.** This is sim-to-sim transfer, not tested on real robots.

---

## 5. Claim supported by this evidence

> A PPO frontier-selection policy trained in a lightweight 2D simulator of the benchmark arena, and transferred zero-shot to a Gazebo/ROS 2 multi-robot SLAM + Nav2 stack, mapped **15 % more area after 240 s** than the hand-designed frontier heuristic (41.8 vs 36.2 m², better in 8/8 paired runs, exact paired permutation p = 0.008), and reached 30 m² **30 % sooner** (121 vs 173 s, p = 0.031) on the benchmark arena.

---

## 6. Reproduce

From `final-product/`, inside the ROS 2 environment:
```bash
colcon build --symlink-install
python3 rl_sim/train.py                                        # ~25 min, writes ~/rl_sim_runs/ppo_<time>/
python3 rl_sim/evaluate.py --model models/ppo_frontier_policy.zip   # 2D comparison
for i in $(seq 8); do                                          # ~90 min: 8 alternating Gazebo pairs
  rl_sim/gazebo_test.sh ppo 240 models/ppo_frontier_policy.zip
  rl_sim/gazebo_test.sh heuristic 240
done
python3 rl_sim/analyze_gazebo_tests.py                         # statistics (section 2.2)
python3 rl_sim/make_graphs.py                                  # all charts in graphs/
```
Raw data: [`results/gazebo_runs/`](../results/gazebo_runs/) (known area per second, one CSV per run), [`results/training/eval_history.csv`](../results/training/), and [`results/sim2d/policy_comparison.csv`](../results/sim2d/).
