# The RL Brain, End to End

This document explains the reinforcement-learning (RL) part of the project from first principles to the last line of code. After reading it you should be able to answer, for every number the policy sees and every number it outputs: *where it comes from, why it is there, and what happens to it next*.

It is written to be read top to bottom, but each part stands on its own:

| Part | What it covers |
|---|---|
| [1. The problem](#1-the-problem-we-are-solving) | What decision the RL agent makes, and why that decision matters |
| [2. RL theory you need](#2-rl-theory-from-zero) | MDPs, returns, value functions, policy gradients, actor-critic, advantage, GAE |
| [3. Our problem as an MDP](#3-our-problem-as-an-mdp) | Exactly what a state, action, reward, step and episode are in our code |
| [4. The 2D simulator](#4-the-2d-training-simulator-rl_sim) | How `rl_sim` builds the arena, scans, finds frontiers, plans and moves |
| [5. The observation](#5-the-observation-153-numbers) | All 153 numbers, one by one |
| [6. Action masking](#6-action-masking) | Why invalid actions are removed and how MaskablePPO does it |
| [7. PPO in depth](#7-ppo-in-depth) | The algorithm, every loss term, every hyperparameter with our numbers |
| [8. The neural network](#8-the-neural-network) | The exact architecture, 74,125 parameters |
| [9. Training, step by step](#9-the-training-run-step-by-step) | What `train.py` does from launch to saved model |
| [10. Reading the training curves](#10-reading-the-training-curves) | What each TensorBoard metric means and what ours did |
| [11. A real episode, traced](#11-a-real-episode-traced-decision-by-decision) | Every decision of the shipped policy, with real numbers |
| [12. Evaluation in 2D](#12-evaluation-in-the-2d-simulator) | `evaluate.py`, the baselines, and why return = speed |
| [13. Deployment to Gazebo](#13-deployment-ros_policy_nodepy) | How `ros_policy_node.py` runs the policy on the real ROS 2 stack |
| [14. Benchmarking and statistics](#14-the-gazebo-benchmark-and-its-statistics) | `gazebo_test.sh`, pairing, the exact permutation test |
| [15. Results and what was learned](#15-results-and-what-the-policy-learned) | The numbers and the behaviour behind them |
| [16. Design decisions](#16-design-decisions-and-why-they-were-made) | Why each choice was made, and what failed before |
| [17. Limitations](#17-limitations) | What the evidence does and doesn't support |
| [18. How to change things](#18-how-to-change-things) | Adding a feature, changing the reward, retraining |
| [19. Glossary](#19-glossary) | Every term in one place |
| [20. Command reference](#20-command-reference) | Every command |

All file paths are relative to `final-product/`.

---

## 1. The problem we are solving

### 1.1 Multi-robot frontier exploration
Two TurtleBot3 robots start in an unknown 8 × 8 m arena with 8–15 box and cylinder obstacles. Each robot has a 360° lidar with 3.5 m range. Each runs SLAM (`slam_toolbox`) to build a map, the two maps are merged into one (`/map`), and each robot drives using Nav2. The goal is to **map the whole arena as fast as possible**.

The map is an *occupancy grid*: a 2D array of cells, each 5 × 5 cm, each either **unknown** (−1), **free** (0) or **occupied** (100).

A **frontier** is the boundary between known free space and unknown space. Standing at a frontier and scanning reveals new area. Frontier-based exploration is a loop:

```
repeat:
    find all frontiers in the current map
    for each robot that needs a goal: choose one frontier
    drive there (Nav2), mapping on the way (SLAM)
until no frontiers are left
```

### 1.2 The one decision the RL agent makes
Everything in that loop is fixed engineering (frontier detection, SLAM, merging, Nav2) **except one line: "choose one frontier"**. That is the only thing the RL agent decides.

The hand-designed baseline (`multi_robot_exploration/frontier_coordinator.py`) makes this choice with a formula:

$$\text{cost} = d + \frac{50}{d_g} + 50 \cdot \max(0,\ x)$$

- $d$: distance from the robot to the frontier
- $d_g$: distance from the frontier to the other robot's current goal (keeps robots apart)
- $x$: how far, in metres, the frontier lies in the *other* robot's half of the arena (splits the arena between robots)

The heuristic takes the frontier with the lowest cost. The weights (50, 50) were picked by hand. The question this project asks is: **can a learned policy make this choice better than the formula?**

### 1.3 Why this choice matters
A bad choice wastes time in two ways: the robot drives far for little new area, or both robots end up mapping the same region. A good choice keeps the robots spread out and picks frontiers whose surroundings reveal a lot at once. The formula captures some of this, but with fixed weights and only three terms. A policy can look at many more features and learn how to trade them off from experience.

---

## 2. RL theory from zero

This section covers exactly the theory our implementation uses, and nothing more.

### 2.1 Agent, environment, and the loop
Reinforcement learning is learning by trial and error. An **agent** interacts with an **environment** in discrete steps:

```
        observation o_t, reward r_t
   ┌───────────────────────────────────┐
   │                                   │
   ▼                                   │
 AGENT ── action a_t ──►  ENVIRONMENT ─┘
```

At each step $t$ the agent sees an observation $o_t$, picks an action $a_t$, and the environment returns a reward $r_{t+1}$ and the next observation $o_{t+1}$. The agent's goal is to collect as much reward as possible over time.

In our project:
- **Agent** = the neural network that picks frontiers.
- **Environment** = the 2D simulator during training (`FrontierExplorationEnv`), and the Gazebo/ROS 2 stack during deployment.
- **Step** = one frontier decision (not one motor command, not one second; see [3.2](#32-what-one-step-is)).

### 2.2 Markov Decision Process (MDP)
Formally, the environment is a **Markov Decision Process**, a tuple $(S, A, P, R, \gamma)$:

| Symbol | Meaning | In our project |
|---|---|---|
| $S$ | set of states | the full simulator state: true map, belief map, robot positions, goals, paths, time |
| $A$ | set of actions | $\{0, 1, \dots, 11\}$: which of the 12 candidate frontier slots to take |
| $P(s' \mid s, a)$ | transition probabilities | the simulator: drive, scan, update the map until the next decision |
| $R(s, a, s')$ | reward | $+0.1$ per newly mapped m², $-0.01$ per simulated second |
| $\gamma$ | discount factor | 0.99 |

**Markov** means the future depends only on the current state, not on how you got there. Our simulator is fully determined by its current state, so that holds.

The agent doesn't see the full state, only a 153-number summary of it (the **observation**). Strictly speaking that makes it a *partially observable* MDP. In practice the observation includes what matters for the choice (positions, goals, frontier properties, elapsed time), so we treat it as if it were the state.

### 2.3 Policy
A **policy** $\pi$ is the agent's behaviour: a mapping from observation to action. Ours is **stochastic** during training: $\pi_\theta(a \mid o)$ is a probability distribution over the 12 slots, computed by a neural network with parameters $\theta$. During deployment we use it **deterministically**: we take the most likely action, $\arg\max_a \pi_\theta(a \mid o)$.

Stochastic during training is essential: it's how the agent *explores* alternatives. If it always took the same action it would never discover a better one.

### 2.4 Return and discounting
The **return** from step $t$ is the sum of future rewards, each discounted by $\gamma$ per step:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

$\gamma < 1$ makes near rewards count more than far ones and keeps the sum finite. With $\gamma = 0.99$ and episodes of only ~7–10 decisions, $\gamma^{7} \approx 0.93$, so in our case discounting barely matters. The return is almost the plain sum of rewards. (The time pressure comes from the explicit $-0.01$ per second penalty, not from discounting.)

The agent's objective is to find parameters $\theta$ that maximise the **expected return**:

$$J(\theta) = \mathbb{E}_{\pi_\theta}\left[ G_0 \right]$$

### 2.5 Value functions
Two functions summarise "how good" things are under a policy $\pi$:

- **State value** $V^\pi(s)$: expected return starting from state $s$ and following $\pi$.
  $$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$$
- **Action value** $Q^\pi(s, a)$: expected return if you take action $a$ in $s$, then follow $\pi$.
  $$Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$$

They are related by the **Bellman equation**: the value of a state is the immediate reward plus the discounted value of where you land.

$$V^\pi(s) = \mathbb{E}_{a \sim \pi,\ s' \sim P}\left[ r + \gamma V^\pi(s') \right]$$

### 2.6 Advantage
The **advantage** says how much better action $a$ is than the policy's average behaviour in state $s$:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

- $A > 0$: this action was better than usual, so make it more likely.
- $A < 0$: worse than usual, so make it less likely.

Advantage is the central quantity in the algorithm we use.

### 2.7 Policy gradient
We want to change $\theta$ to increase $J(\theta)$. The **policy gradient theorem** gives the gradient in a form we can estimate from experience:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A^{\pi}(s_t, a_t) \right]$$

In words: for every action taken, push up its log-probability in proportion to its advantage. Good actions become more likely, bad ones less likely. This is "trial and error" made precise.

We don't know the true $A^\pi$, so we estimate it. That's what the critic is for.

### 2.8 Actor-critic
An **actor-critic** method learns two things at once:

- the **actor** $\pi_\theta(a \mid s)$: the policy that picks actions;
- the **critic** $V_\phi(s)$: an estimate of the state value, used to compute advantages.

The critic is trained like a regression: make $V_\phi(s_t)$ match the observed returns. The actor uses the critic to judge its actions. Our network has a separate actor and critic, each a 2 × 128 MLP ([section 8](#8-the-neural-network)).

### 2.9 Temporal-difference error and GAE
The simplest advantage estimate uses one step of real reward plus the critic's guess for the rest. It's called the **TD error**:

$$\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

That has low variance but is biased if the critic is wrong. Using the full observed return instead is unbiased but noisy. **Generalised Advantage Estimation (GAE)** blends the two with a parameter $\lambda$:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l\, \delta_{t+l}$$

- $\lambda = 0$: one-step TD (low variance, more bias).
- $\lambda = 1$: full Monte-Carlo return minus baseline (no bias, high variance).
- We use $\lambda = 0.95$, the standard compromise.

In practice SB3 computes this backwards over each rollout buffer, resetting at episode ends:

```
last_gae = 0
for t from last to first:
    next_value = 0 if episode ended at t else V(s_{t+1})
    delta = r_t + γ · next_value − V(s_t)
    last_gae = delta + γ · λ · (0 if episode ended at t else last_gae)
    advantage[t] = last_gae
returns[t] = advantage[t] + V(s_t)        # targets for the critic
```

### 2.10 Why plain policy gradient is fragile, and what PPO fixes
Plain policy gradient takes one gradient step per batch of experience, then throws the batch away. If the step is too large, the policy can change drastically and collapse, and the data you collected no longer describes the new policy. **Proximal Policy Optimization (PPO)** lets you reuse each batch for several epochs while preventing any single update from moving the policy too far. [Section 7](#7-ppo-in-depth) covers it in full.

### 2.11 On-policy vs off-policy
PPO is **on-policy**: it learns only from experience collected by the current policy, then discards it. (Off-policy methods like DQN and SAC keep a replay buffer of old experience.) On-policy methods are simpler and more stable, but need more environment steps. That's affordable here because our simulator is fast (~0.1 s per episode).

---

## 3. Our problem as an MDP

The environment is `FrontierExplorationEnv` in `rl_sim/envs/frontier_env.py`. It follows the standard **Gymnasium** API (`reset()`, `step(action)`, `observation_space`, `action_space`), which is what Stable-Baselines3 expects.

### 3.1 Spaces
```python
self.observation_space = spaces.Box(-5.0, 5.0, (153,), np.float32)   # 153 real numbers
self.action_space      = spaces.Discrete(12)                          # slot index 0..11
```

### 3.2 What one step is
This is the most important design decision in the whole RL setup. **One step = one frontier decision for one robot.** It is *not* one control tick or one second.

After the agent picks a frontier, the simulator runs as many 0.5 s ticks as needed (robots drive, lidars scan, the map grows) until **some robot needs a new goal**. Only then does it ask the agent again. So:

- one step can last 0 s or 20+ s of simulated time;
- an episode is only **~7–10 decisions** long, not thousands of motor commands.

In RL terms, each action is a temporally extended **option** ("go to this frontier"), and the problem is a **semi-MDP**. Short episodes make credit assignment easy: the reward for a good choice arrives within a few steps of it.

### 3.3 Two robots, one policy (turn-taking)
There are two robots but **one policy**. Whenever a robot is idle, the policy is asked to choose for *that* robot, seeing the world from that robot's point of view (its position is "ego", the other robot is "other"). This is called **parameter sharing**: both robots run the same network.

If both robots are idle at the same moment, they take turns, and the order alternates (`self.turn`) so neither robot always goes first. The second robot's decision already sees the first robot's new goal, so it can avoid it. That is why, in the traced episode ([section 11](#11-a-real-episode-traced-decision-by-decision)), the first step has 0 s duration: robot 1 decided, then robot 2 was also idle and was asked immediately.

### 3.4 `reset()`
```
belief map ← all unknown (−1)
place both robots at their spawn cells (from the arena generator)
both robots scan once                      → ~35.7 m² known at t = 0
t ← 0, decisions ← 0, blacklist ← {}
prev_known ← current known area
_advance()                                 → runs until the first decision is needed
return the first observation
```
Every episode starts on the benchmark arena with the same spawn positions, so the environment is **deterministic**: given the same actions, it produces exactly the same episode. All randomness during training comes from the policy's sampling.

### 3.5 `step(action)`
```
(robot i, obs, mask, candidates) ← the pending decision
if the action is masked: replace it with the first valid slot     # safety net, never used by MaskablePPO
plan an A* path from robot i to candidates[action]; set it as robot i's goal
decisions += 1
t0 ← t
_advance()                       # simulate until the next decision or the end
reward ← 0.1 · (known area now − known area before) − 0.01 · (t − t0)
terminated ← (end == 'explored')
truncated  ← (end == 'time_limit')     # 600 s or 400 decisions, never reached in practice
return next obs, reward, terminated, truncated, info
```

### 3.6 `_advance()`: the inner simulation loop
```
loop:
    if t ≥ 600 s: episode ends (time_limit)
    idle ← robots without a goal
    if any idle:
        frontiers ← detect_frontiers(belief)
        for each idle robot (alternating order):
            build its decision (candidates, observation, mask)
            if it has at least one candidate: store it as pending, return  → the agent is asked
        if all robots are idle and none has a candidate: episode ends (explored)
    move every robot with a goal by one 0.5 s tick and scan
    t += 0.5 s
```
Note the subtle end condition: a robot with no candidates simply stays idle while the other keeps working. The episode only ends when **nobody** has anything useful left to do.

### 3.7 Reward
```python
AREA_REWARD  = 0.1    # per m² newly known
TIME_PENALTY = 0.01   # per simulated second
reward = AREA_REWARD * (known - prev_known) - TIME_PENALTY * (t - t0)
```

**Why this reward makes the agent fast.** Every finished episode maps the same area: everything reachable, which on the benchmark arena is 35.66 → 60.46 m², i.e. **24.8 m²** of new area. So the total return of any finished episode is

$$G = 0.1 \times 24.8 - 0.01 \times T = 2.48 - 0.01\,T$$

where $T$ is the time to finish. **Maximising return is exactly minimising exploration time.** You can check this against the real results:

| Policy | Time $T$ | $2.48 - 0.01\,T$ | Measured return |
|---|---|---|---|
| PPO | 67 s | 1.81 | 1.8098 |
| Heuristic | 111 s | 1.37 | 1.3748 |
| Nearest | 112 s | 1.36 | 1.3613 |
| Random | 163 s | 0.85 | 0.8558 |

So why include the area term at all, if it adds up to a constant? Because it arrives **during** the episode. It tells the agent *which* decisions produced new area, and *when*. Without it, the only signal would be the total time at the end, which is much harder to attribute to individual choices. This is a form of **reward shaping**.

**Why these scales.** Per-step rewards land between about −0.2 and +0.8, and returns around 1–2. Neural networks learn best when the values they predict are of order 1, so no reward normalisation is needed.

### 3.8 Episode end
| `end_reason` | When | Gymnasium flag |
|---|---|---|
| `explored` | all robots idle and no robot has a valid candidate | `terminated = True` |
| `time_limit` | 600 s simulated, or 400 decisions (safety caps) | `truncated = True` |

The distinction matters to PPO: after `terminated` the future value is 0. After `truncated`, the episode was cut short artificially, so in principle the critic's estimate should be used. In our runs, episodes always end with `explored`, well before either cap.

---

## 4. The 2D training simulator (`rl_sim`)

Training in Gazebo would take ~5 minutes per episode. `rl_sim` runs an episode in ~0.1 s, about **3000× faster**, by keeping only what matters for the frontier decision. Each piece mirrors a piece of the real stack:

| Real stack | rl_sim replacement | File |
|---|---|---|
| Gazebo world | the same world file, rasterised at 0.05 m | `core/world.py` |
| LDS-01 lidar + slam_toolbox | ideal 360-ray raycaster writing into a shared belief grid | `core/lidar.py` |
| merged `/map` | the shared belief grid itself | `envs/frontier_env.py` |
| `frontier_coordinator` frontier detection | the identical algorithm and parameters | `core/frontiers.py` |
| Nav2 planning + driving | A* on an inflated map, 0.15 m/s | `core/planner.py`, `_move()` |

### 4.1 The arena (`core/world.py`)
1. `generate_random_world()`, the same function Gazebo's launch file calls, writes the SDF world file of the benchmark arena (arena 42, which Gazebo selects with `GAZEBO_WORLD_SEED=42`) and returns the two spawn positions.
2. `_rasterise()` parses the SDF with `xml.etree`, and for every model's collision geometry draws boxes as filled rectangles and cylinders as filled circles onto a 170 × 170 grid (8.5 m × 8.5 m at 0.05 m per cell, covering the 8 m arena plus walls). Cells are `FREE = 0` or `OCCUPIED = 100`. This is the **ground truth**.
3. **Inflation:** the occupied cells are dilated by 0.30 m (robot radius 0.105 m plus a margin, consistent with Nav2's 0.35 m inflation). Everything not covered is `nav_free`: where the robot's *centre* may be. Paths and goals stay in `nav_free`, so they keep the same clearance Nav2 would.
4. **Connected components** of `nav_free` are labelled (8-connectivity). Two cells with the same label are mutually reachable. This lets the environment reject unreachable frontiers instantly, without running A*.
5. Spawn positions are converted to cells, and nudged to the nearest `nav_free` cell if needed.

The result is cached, so all 8 training processes build it once each.

**Coordinates.** Cell `(row, col)` has world centre `x = −4.25 + (col + 0.5)·0.05`, `y = −4.25 + (row + 0.5)·0.05`. Row is y, column is x.

### 4.2 The lidar (`core/lidar.py`)
A vectorised ray-caster for 360 rays at 1° spacing, out to 3.5 m (70 cells):
- At construction it precomputes, for every ray and every step along it, the row/column offset. A step of 0.999 cells ensures no cell is skipped.
- `scan()` looks up the ground truth along all rays at once (NumPy fancy indexing), finds the first occupied cell on each ray (`argmax` of the hit mask), and **copies the ground truth into the belief** for every cell up to and including that hit.

This is an *ideal* sensor: no noise, no missed returns, no SLAM drift. The belief is exactly correct wherever a ray has reached. That is the main source of the sim-to-sim gap ([section 13.6](#136-the-sim-to-sim-gap)).

### 4.3 Frontier detection (`core/frontiers.py`)
This is a line-by-line port of `frontier_coordinator._detect_frontiers()` and `_deduplicate()`, so that both systems see exactly the same frontiers.

```
unknown  = belief == −1
free     = belief == 0
obstacle = belief ≥ 50
obstacle = morphological opening (3×3)          # remove single-pixel noise
dilated_obstacles = dilate(obstacle, 7×7 ellipse)  # ~0.15 m safety margin
safe_free = free AND NOT dilated_obstacles
frontier_mask = safe_free AND dilate(unknown, 3×3)   # free cells touching unknown
components = connected components of frontier_mask
keep components with more than 15 pixels; record centroid and size
dedup: walk the list, keep a centroid only if it is > 1.5 m from every kept one
```
The output is a list of frontier **centroids** (row, col) and **sizes** (pixel count).

### 4.4 Path planning (`core/planner.py`)
Standard **A\*** on the 8-connected `nav_free` grid. Straight moves cost 1 and diagonal moves cost √2. The heuristic is the *octile distance* $\max(\Delta r, \Delta c) + (\sqrt{2} - 1)\min(\Delta r, \Delta c)$, which is exact on an 8-connected grid without obstacles, so A\* stays optimal. It stands in for Nav2's global planner.

### 4.5 Motion (`_move()`)
Each 0.5 s tick, a robot advances `0.15 m/s × 0.5 s = 0.075 m = 1.5 cells` along its path, possibly passing several waypoints in one tick. Its heading is updated to the direction of motion. It scans at its new cell. Its goal **ends** when either:
- the path is finished, or
- the 1 m window around the goal is fully known (`_goal_explored`): the frontier has "vanished", so there is no point continuing.

The second rule mirrors the coordinator, which abandons a goal when its frontier disappears. It is also used, unchanged, in deployment.

### 4.6 Turning frontiers into candidates (`_candidates()`)
Not every frontier centroid is a good goal. The centroid may sit in an obstacle's inflation, or be unreachable. For robot $i$:
1. **Snap** each centroid to the nearest `nav_free` cell within 0.6 m that is in the **same connected component** as the robot. If there is none, the frontier is unreachable: skip it.
2. Skip it if the snapped goal is **blacklisted** (failed in the last 30 s) or its surroundings are already fully known.
3. Skip it if it is closer than **0.6 m** to the robot (pointless trip; same rule as the coordinator).
4. Skip it if it is within **1.5 m of the other robot's goal** (the other robot will map it anyway).
5. Sort the survivors by straight-line distance and keep the **12 nearest**.

These 12 (or fewer) are the **candidates**. Slot 0 is the nearest, slot 1 the next, and so on. This ordering gives the slots a stable meaning: "slot 0" always means "the nearest valid frontier".

---

## 5. The observation: 153 numbers

The observation is built in `_decision()` as **12 candidate rows × 12 features + 9 global features = 153**. Features are hand-scaled to roughly [−1, 1] or [0, 3] so the network sees well-conditioned inputs. No running normalisation (`VecNormalize`) is used.

### 5.1 Per-candidate features (12 per slot)
Positions are in world metres divided by `ARENA = 4.0`, so the arena spans about [−1, 1]. Distances are divided by 8 m.

| # | Feature | Formula | Why it's there |
|---|---|---|---|
| 0 | valid flag | 1.0 for a real candidate, 0 for an empty slot | tells the network which rows are padding |
| 1 | goal x | $x_g / 4$ | where the frontier is in the arena |
| 2 | goal y | $y_g / 4$ | |
| 3 | offset x | $(x_g - x_{ego}) / 4$ | where it is *relative to me* (ego-centric) |
| 4 | offset y | $(y_g - y_{ego}) / 4$ | |
| 5 | distance | $d / 8$ | travel cost |
| 6 | distance to other robot | $\|g - p_{other}\| / 8$ | avoid crowding the other robot |
| 7 | distance to other's goal | $\|g - g_{other}\| / 8$, or 1.5 if it has none | avoid duplicating its work |
| 8 | frontier size | $\min(\text{pixels} / 100,\ 3)$ | bigger frontiers usually mean more unknown behind them |
| 9 | unknown fraction | fraction of unknown cells in the 2 m × 2 m window around the goal | **information gain**: how much a visit is likely to reveal |
| 10 | heading alignment | $\cos(\text{bearing to goal} - \text{robot heading})$ | +1 straight ahead, −1 behind (turning costs time in reality) |
| 11 | heuristic's pick | 1.0 on the slot the hand-designed formula would choose | gives the network the baseline's opinion as a hint |

Feature 9 is computed for the whole map at once with a box filter (`cv2.boxFilter` of the unknown mask, window 41 × 41 cells ≈ 2 m), then read at each goal cell.

Feature 11 deserves a comment. It makes "do what the heuristic does" trivially easy to learn, which gives the policy a strong starting point. The policy is free to deviate whenever the other features say it should. In the traced episode it agrees with the heuristic on 4 of 7 decisions and disagrees on 3.

### 5.2 Global features (9)
| # | Feature | Formula |
|---|---|---|
| 0–1 | my position | $(x_{ego}, y_{ego}) / 4$ |
| 2–3 | other robot's position | $(x_{other}, y_{other}) / 4$ |
| 4 | other robot has a goal | 1.0 or 0.0 |
| 5–6 | other robot's goal | $(x, y) / 4$, or (0, 0) if none |
| 7 | explored fraction | known m² / free m² of the arena |
| 8 | elapsed time | $t / 600$ s |

### 5.3 Empty slots
If there are fewer than 12 candidates, the remaining rows stay all zeros (including the valid flag), and those slots are masked out of the action ([section 6](#6-action-masking)).

### 5.4 A real observation
The first decision of an episode on the benchmark arena: robot 1 at (0.42, −2.58) m, 8 valid candidates. The rows (rounded) are:

```
slot valid   x     y    dx    dy   dist d_oth d_ogoal size  unk  cos  heur   world goal (m)
 0   1.0   0.37 -0.32  0.26  0.32  0.21  0.26  1.5   0.16 0.09  0.63  1.0   ( 1.48, -1.27)
 1   1.0   0.46 -0.88  0.35 -0.24  0.21  0.48  1.5   0.19 0.26  0.83  0.0   ( 1.83, -3.52)
 2   1.0  -0.54 -0.88 -0.65 -0.24  0.35  0.79  1.5   2.12 0.33 -0.94  0.0   (-2.17, -3.52)
 3   1.0  -0.56 -0.36 -0.66  0.29  0.36  0.67  1.5   3.0  0.36 -0.92  0.0   (-2.23, -1.42)
 4   1.0  -0.26 -0.02 -0.36  0.62  0.36  0.49  1.5   3.0  0.48 -0.5   0.0   (-1.02, -0.08)
 5   1.0   0.83 -0.79  0.73 -0.15  0.37  0.42  1.5   3.0  0.36  0.98  0.0   ( 3.33, -3.17)
 6   1.0   0.19  0.48  0.09  1.12  0.56  0.35  1.5   3.0  0.28  0.08  0.0   ( 0.78,  1.93)
 7   1.0   0.86  0.76  0.75  1.4   0.79  0.36  1.5   2.67 0.43  0.47  0.0   ( 3.43,  3.03)
 8–11: all zeros (masked)
global: [0.106, -0.644, 0.731, 0.044, 0.0, 0.0, 0.0, 0.567, 0.0]
```

Reading it: the heuristic picks slot 0 (nearest, 1.7 m away, but only 9 % unknown around it). The trained policy gives **86 %** probability to slot 1, which is just as near but has **26 %** unknown around it and lies almost straight ahead (cos = 0.83). It gives 12 % to slot 5. The policy has learned to weigh information gain more than the formula does.

---

## 6. Action masking

### 6.1 The problem
The action space is always `Discrete(12)`, but often only 2–8 slots hold a real candidate. Without masking, the agent would sometimes pick an empty slot. You'd then have to either penalise that (wasting experience on learning "don't pick empty slots") or silently remap it (making the action's effect inconsistent).

### 6.2 The solution
**Invalid action masking** removes invalid actions from the distribution before sampling. The environment exposes `action_masks()`, a boolean array of length 12 (`True` for the first `len(candidates)` slots). **MaskablePPO** from `sb3-contrib` calls it at every step.

Inside the policy, the network outputs 12 **logits** $z_1 \dots z_{12}$ (unnormalised scores). Masked logits are replaced by a huge negative number before the softmax:

$$z'_k = \begin{cases} z_k & \text{if slot } k \text{ is valid} \\ -10^8 & \text{otherwise} \end{cases} \qquad \pi(k \mid o) = \frac{e^{z'_k}}{\sum_j e^{z'_j}}$$

so invalid slots get probability exactly 0. Three consequences:
- **Sampling** never picks an invalid slot.
- **Log-probabilities and gradients** only involve valid slots. No learning signal is wasted on impossible actions.
- **Entropy** is computed over valid slots only. With ~3–4 valid slots on average, the maximum entropy is about $\ln 3.7 \approx 1.3$, which is exactly where our training started ([section 10](#10-reading-the-training-curves)).

The mask is also passed at deployment: `model.predict(obs, action_masks=mask, deterministic=True)`.

### 6.3 Masking vs the features
The masked slots' features are also zero, so the network *could* infer validity from feature 0. The mask makes it a hard guarantee rather than something to learn.

---

## 7. PPO in depth

### 7.1 The algorithm in one picture
```
initialise actor π_θ and critic V_φ randomly
repeat until 300,000 decisions have been collected:
    ── COLLECT (rollout) ─────────────────────────────────────────────
    run the current policy in 8 environments for 256 steps each
        → 2,048 transitions (o, a, r, done, log π_old(a|o), V(o), mask)
    ── ESTIMATE ──────────────────────────────────────────────────────
    compute advantages Â with GAE (γ = 0.99, λ = 0.95)
    compute critic targets  R̂ = Â + V(o)
    ── OPTIMISE ──────────────────────────────────────────────────────
    for 10 epochs:
        shuffle the 2,048 transitions into 8 minibatches of 256
        for each minibatch:
            normalise Â within the minibatch (mean 0, std 1)
            compute the loss L (below) and take one Adam step
    discard the data
```
With our settings: **2,048** transitions per update, **8** minibatches × **10** epochs = **80** gradient steps per update, and 300,000 / 2,048 ≈ **146** updates in total.

### 7.2 The probability ratio
For each stored transition, PPO compares the new policy with the one that collected the data:

$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_\text{old}}(a_t \mid o_t)} = \exp\left(\log \pi_\theta(a_t \mid o_t) - \log \pi_{\theta_\text{old}}(a_t \mid o_t)\right)$$

$\rho = 1$ means unchanged, $\rho = 1.5$ means the action became 50 % more likely.

### 7.3 The clipped surrogate objective
The core of PPO:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[ \min\left( \rho_t \hat{A}_t,\ \ \text{clip}(\rho_t,\ 1 - \epsilon,\ 1 + \epsilon)\, \hat{A}_t \right) \right], \qquad \epsilon = 0.2$$

How to read it:
- If $\hat{A}_t > 0$ (good action), increasing $\rho$ increases the objective, **but only until $\rho = 1.2$**. Beyond that the clip makes the gradient zero. You can make a good action up to 20 % more likely per update, no more.
- If $\hat{A}_t < 0$ (bad action), decreasing $\rho$ helps, but only down to $\rho = 0.8$.
- The `min` makes this a *pessimistic* bound. It never rewards moving far from the old policy, but it always penalises moving in the wrong direction.

This is what makes it safe to reuse each batch for 10 epochs. Once the policy has moved 20 % on an action, that sample stops pushing.

### 7.4 The value loss
The critic is trained to predict the GAE return targets:

$$L^{V}(\phi) = \mathbb{E}_t\left[ \left( V_\phi(o_t) - \hat{R}_t \right)^2 \right]$$

(SB3 doesn't clip the value function by default, and we don't enable it.)

### 7.5 The entropy bonus
The **entropy** of the policy at a state measures how spread out its choices are:

$$H(\pi(\cdot \mid o)) = -\sum_k \pi(k \mid o) \log \pi(k \mid o)$$

Adding entropy to the objective rewards keeping options open. It slows premature collapse onto one action and keeps exploration going early in training.

### 7.6 The total loss
SB3 minimises

$$L = -L^{\text{CLIP}} + c_v \, L^{V} - c_e \, H, \qquad c_v = 0.5,\ c_e = 0.01$$

with one Adam optimiser over all parameters. Gradients are clipped to a global norm of 0.5 before each step (`max_grad_norm`), which protects against occasional huge gradients.

### 7.7 Every hyperparameter, explained
From `rl_sim/train.py`:

```python
model = MaskablePPO(
    'MlpPolicy', venv,
    n_steps=256, batch_size=256, n_epochs=10,
    learning_rate=lambda progress: 3e-4 * progress,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    tensorboard_log=..., seed=args.seed, verbose=0)
```

| Parameter | Value | What it controls | Why this value |
|---|---|---|---|
| `n_envs` | 8 | parallel simulators | 8 CPU cores; more diverse data per update |
| `n_steps` | 256 | steps per env per rollout | 8 × 256 = 2,048 transitions ≈ 290 episodes per update: large, low-variance batches |
| `batch_size` | 256 | minibatch size | 8 minibatches per epoch |
| `n_epochs` | 10 | passes over each rollout | reuse the data; the clip keeps it safe |
| `learning_rate` | 3e-4 → 0, linear | Adam step size | `progress` is SB3's "progress remaining", going 1 → 0, so the lr decays linearly to 0 and the policy settles at the end |
| `gamma` | 0.99 | discount | near-undiscounted; episodes are short |
| `gae_lambda` | 0.95 | GAE bias/variance trade-off | standard |
| `clip_range` | 0.2 | PPO's $\epsilon$ | standard |
| `ent_coef` | 0.01 | entropy bonus weight | keeps trying different slots early on |
| `vf_coef` | 0.5 | value-loss weight | standard |
| `max_grad_norm` | 0.5 | gradient clipping | standard |
| `net_arch` | pi: [128, 128], vf: [128, 128] | network size | ~74k parameters; plenty for 153 inputs |
| total timesteps | 300,000 decisions | training length | converges by ~30–60k; the rest confirms stability |

### 7.8 How MaskablePPO differs from PPO
Only in the policy distribution. `sb3-contrib`'s `MaskableCategorical` applies the mask to the logits ([6.2](#62-the-solution)) whenever it samples, evaluates log-probabilities or computes entropy. The rollout buffer stores the mask of each transition, so the optimisation epochs recompute log-probabilities under the same mask that was active when the action was taken. Everything else (GAE, clipping, losses) is identical to PPO.

---

## 8. The neural network

`MlpPolicy` with `net_arch=dict(pi=[128, 128], vf=[128, 128])` creates two **separate** networks with no shared layers. Printed from the shipped model:

```
MaskableActorCriticPolicy(
  features_extractor: Flatten                         # 153 → 153 (no-op for a vector)
  mlp_extractor:
    policy_net: Linear(153→128) → Tanh → Linear(128→128) → Tanh
    value_net:  Linear(153→128) → Tanh → Linear(128→128) → Tanh
  action_net: Linear(128→12)       # 12 logits, one per slot
  value_net:  Linear(128→1)        # V(o)
)
```

```
                 ┌─ Linear 153→128 ─ tanh ─ Linear 128→128 ─ tanh ─ Linear 128→12 ─ mask ─ softmax → π(slot | o)
 obs (153) ──────┤                                                                          (ACTOR)
                 └─ Linear 153→128 ─ tanh ─ Linear 128→128 ─ tanh ─ Linear 128→1  ───────────→ V(o)
                                                                                            (CRITIC)
```

**Parameter count**

| Part | Weights + biases |
|---|---|
| actor hidden | 153·128 + 128 + 128·128 + 128 = 36,224 |
| actor head | 128·12 + 12 = 1,548 |
| critic hidden | 36,224 |
| critic head | 128 + 1 = 129 |
| **total** | **74,125** |

**Details from SB3's defaults:**
- **Activation:** tanh, which keeps hidden activations in [−1, 1].
- **Initialisation:** orthogonal weights. Gain √2 for hidden layers, **0.01 for the action head** (so the initial policy is nearly uniform over valid slots, i.e. maximum exploration), and 1 for the value head.
- **Optimiser:** Adam, ε = 1e-5.

**Why an MLP and not a CNN?** The input is a feature vector, not an image, so there is no spatial structure for convolutions to exploit. The features already summarise the spatial information the decision needs ([16.3](#163-a-feature-vector-not-a-map-image)).

**Size on disk:** `models/ppo_frontier_policy.zip` (~0.9 MB) contains the network weights (`policy.pth`), the Adam optimiser state (`policy.optimizer.pth`), and the hyperparameters (`data`).

---

## 9. The training run, step by step

`python3 rl_sim/train.py` does the following.

1. **Parse arguments:** `--world 42` (benchmark arena), `--timesteps 300000`, `--num-envs 8`, `--eval-every 10000`, `--run-dir ~/rl_sim_runs/ppo_<time>`.
2. **Heuristic baseline:** run one episode with `env.heuristic_action()` and print its return and time (1.375, 111 s). It's used as a reference line in every evaluation.
3. **Vectorised environments:** `SubprocVecEnv` starts 8 `FrontierExplorationEnv` instances in 8 processes. `VecMonitor` records episode returns and lengths (`rollout/ep_rew_mean`, `ep_len_mean`). When an environment's episode ends, the vec-env resets it automatically, so collection never stalls.
4. **Build MaskablePPO** with the settings in [7.7](#77-every-hyperparameter-explained).
5. **`model.learn(300_000, callback=EvalCallback)`:** the loop in [7.1](#71-the-algorithm-in-one-picture), ~146 times.
6. **`EvalCallback`**, every 10,000 decisions:
   - runs one episode in a separate evaluation env with the **deterministic** policy (argmax, masked);
   - logs `eval/return` and `eval/explore_time_s` to TensorBoard and appends a row to `eval_history.csv`;
   - if the return is **strictly better** than any previous evaluation, saves `best_model.zip` and prints `<- best, saved`.

   Since the environment is deterministic, one evaluation episode gives the exact deterministic performance. No averaging is needed.
7. **On exit** (normal or Ctrl-C): save `final_model.zip`, close the environments, export every TensorBoard scalar to `tensorboard_scalars.csv` (`export_tensorboard.py`).

**What the run printed** (from `results/training/eval_history.csv`):

| Decisions | Eval return | Explore time | |
|---|---|---|---|
| 0 | 0.573 | 191.0 s | untrained (worse than random's 163 s on this run) |
| 10,000 | 1.055 | 141.5 s | |
| 20,000 | 1.624 | 86.0 s | already beats the heuristic (111 s) |
| **30,000** | **1.810** | **67.0 s** | **best, saved** |
| 40,000 … 300,000 | 1.810 | 67.0 s | unchanged |

**Which model is shipped.** `models/ppo_frontier_policy.zip` is the `best_model.zip` of this run. Later evaluations only *tied* 1.810 and never beat it strictly, so the best model is the **checkpoint at 30,008 decisions** (its stored `num_timesteps`), about 4 minutes into training. It already has the final deterministic behaviour. Training continued to 300k and sharpened the stochastic policy (entropy → 0) without changing the argmax choices.

Training takes ~25 minutes on a laptop CPU (at 107–209 decisions per second, i.e. ~15–30 episodes per second across 8 processes). No GPU is needed: the network is tiny, and the time goes into simulation.

---

## 10. Reading the training curves

Charts: `graphs/11_…` to `graphs/19_…`, combined in `graphs/20_training_dashboard.svg`. Raw values: `results/training/tensorboard_scalars.csv`.

| Metric | Meaning | Healthy behaviour | Ours (start → 30k → 60k → end) |
|---|---|---|---|
| `rollout/ep_rew_mean` | mean return of recent training episodes (stochastic policy) | rises, then plateaus | 1.07 → 1.69 → 1.81 → 1.81 |
| `rollout/ep_len_mean` | decisions per episode | settles | 7.8 → … → 7.0 |
| `train/entropy_loss` | **minus** the mean policy entropy | rises from −(max entropy) towards 0 as the policy becomes confident | −1.31 → −0.37 → −0.007 → −0.0002 |
| `train/explained_variance` | $1 - \text{Var}(\hat{R} - V) / \text{Var}(\hat{R})$: how much of the return the critic explains | → 1 | −1.15 → 0.98 → 0.999 → 1.0 |
| `train/value_loss` | critic's squared error | falls | 0.175 → … → ~1e-16 |
| `train/policy_gradient_loss` | the clipped surrogate loss | small, noisy, → 0 when the policy stops changing | −0.0125 → … → ~0 |
| `train/approx_kl` | approximate KL divergence between the old and new policy per update: how far one update moved it | small (≈0.01–0.02) while learning, → 0 when converged | 0.016 → 0.021 → 0.0006 → 0 |
| `train/clip_fraction` | fraction of samples where the clip was active | > 0 while learning, → 0 when converged | 0.11 → … → 0 |
| `train/learning_rate` | current lr | linear decay | 3.0e-4 → 1e-6 |
| `eval/return`, `eval/explore_time_s` | deterministic evaluation ([9](#9-the-training-run-step-by-step)) | improves, then flat | 0.57 / 191 s → 1.81 / 67 s from 30k on |

**How to interpret these together:**
- **Start:** entropy −1.31 ≈ $-\ln(3.7)$ means the policy is uniform over the ~3–4 valid slots per decision (the 0.01 head init at work). Explained variance is negative: the untrained critic is worse than predicting the mean.
- **10k–40k:** the learning phase. KL is around 0.01–0.02 and clip fraction is high: every update is moving the policy as much as PPO allows. Entropy drops as the policy commits. Explained variance shoots up: the critic has learned to predict returns.
- **After ~60k:** converged. KL and clip fraction are 0 (updates no longer change anything), entropy is ~0 (fully deterministic), and value loss is ~0. In a deterministic environment with a deterministic policy, every episode is identical, so the critic can predict the return *exactly*. That's why explained variance reaches 1.0, which would be impossible in a noisy environment.

**Warning signs to look for when you change something:** KL repeatedly > 0.05 (lr too high), entropy collapsing before the return improves (premature convergence: raise `ent_coef`), explained variance staying low (critic can't learn: check reward scale and features), or return oscillating (lr too high or batches too small).

---

## 11. A real episode, traced decision by decision

The shipped model on the benchmark arena (2D simulator, deterministic). Robot 1 spawns at (0.42, −2.58) m and robot 2 at (2.93, 0.17) m. 35.7 m² are known after the initial scans.

| # | Robot | Valid slots | PPO picks | Heuristic would pick | Goal (m) | Sim time | Reward | Known m² |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 8 | slot 1 | slot 0 | (1.83, −3.52) | 0.0 → 0.0 s | 0.000 | 35.7 |
| 2 | 2 | 7 | slot 1 | slot 2 | (0.78, 1.93) | 0.0 → 12.0 s | +0.765 | 44.5 |
| 3 | 1 | 4 | slot 0 | slot 0 | (−2.07, −0.82) | 12.0 → 19.5 s | +0.297 | 48.2 |
| 4 | 2 | 4 | slot 0 | slot 1 | (0.53, 3.43) | 19.5 → 33.5 s | +0.547 | 55.1 |
| 5 | 2 | 5 | slot 0 | slot 0 | (−0.32, 2.68) | 33.5 → 43.0 s | +0.330 | 59.4 |
| 6 | 2 | 2 | slot 0 | slot 0 | (−3.52, 2.53) | 43.0 → 45.5 s | +0.027 | 59.9 |
| 7 | 1 | 2 | slot 1 | slot 1 | (−3.52, −3.42) | 45.5 → 67.0 s | −0.157 | 60.5 |
| | | | | | | **67.0 s** | **1.810** | |

What this shows:
- **Decision 1 has zero duration and zero reward.** Both robots were idle at t = 0. Robot 1 chose first, and robot 2 was asked immediately (turn-taking, [3.3](#33-two-robots-one-policy-turn-taking)).
- **Rewards check out:** 0.1 × (44.5 − 35.7) − 0.01 × 12.0 = 0.88 − 0.12 ≈ 0.765 for decision 2 (using unrounded areas).
- **The last decision has negative reward.** Robot 1 drives 21.5 s to a far corner for only 0.6 m². That trip is still worth it, because the episode can't end (and the time penalty can't stop) until the corner is mapped.
- **PPO disagrees with the heuristic on 3 of 7 decisions:** the first decision of each robot and robot 2's second. These early choices set up the whole episode, which then finishes in 7 decisions instead of the heuristic's 9.
- **The critic's view:** at decision 1 the critic predicts $V = 1.64$. The actual return is 1.81. The 30k checkpoint's critic wasn't perfectly converged yet (explained variance 0.98 at that point).

---

## 12. Evaluation in the 2D simulator

`python3 rl_sim/evaluate.py --model models/ppo_frontier_policy.zip` runs one deterministic episode per policy and saves the final map of each (`--video` also records an MP4):

| Policy | How it chooses | Return | Time to explore | Decisions |
|---|---|---|---|---|
| **PPO** | the trained network, argmax | **1.810** | **67.0 s** | 7 |
| Heuristic | lowest `frontier_coordinator` cost | 1.375 | 111.0 s | 9 |
| Nearest | slot 0 (candidates are sorted by distance) | 1.361 | 112.0 s | 9 |
| Random | uniformly among valid slots | 0.856 | 163.0 s | 9 |

Stored in `results/sim2d/policy_comparison.csv` and charted in `graphs/03_sim2d_time_to_explore.svg`. Because return = 2.48 − 0.01 × time ([3.7](#37-reward)), the return and time columns tell the same story. PPO explores the arena **40 % faster** than the heuristic in 2D.

Note that the heuristic is barely better than "nearest". Its separation and region terms hardly help on this arena, which leaves room for a learned policy.

---

## 13. Deployment: `ros_policy_node.py`

Training happens in 2D. The test is whether the same network, **unchanged and without further training** ("zero-shot"), still helps on the full Gazebo + SLAM + Nav2 stack. This is **sim-to-sim transfer** (2D simulator → 3D physics simulator).

### 13.1 The key idea: reuse the training code for the observation
The most common way sim-to-real transfer silently fails is that the deployed system computes the observation slightly differently from training: a different scaling, a swapped axis, a feature computed from a different map. Here, the node **holds a `FrontierExplorationEnv` instance** and *overwrites its internal state* with the real state every second. It then calls the env's own `_decision()` to build the observation and mask. The 153 numbers are produced by exactly the same code in both worlds.

### 13.2 Inputs from ROS 2
| Input | Source | Used for |
|---|---|---|
| `/map` (`nav_msgs/OccupancyGrid`) | `map_merge_node`, 2 Hz, transient-local QoS | the belief grid |
| robot pose | TF lookup `map → robotN/base_footprint` | position (and yaw from the quaternion) |
| Nav2 goal status | `NavigateToPose` action responses and results | when a goal ends |
| sim time | `/clock` (`use_sim_time = True`) | `env.t`, timeouts, blacklist |

### 13.3 The 1-second tick (`_tick()`), step by step
```
1. wait until both Nav2 action servers are up, /map has arrived, and both TF poses resolve
2. first tick only: record the start time; set each robot's "home" (used by the heuristic's
   region split) to its starting cell
3. env.t ← sim time since start
4. env.belief ← _resample_map(/map)
5. drop blacklist entries older than 30 s
6. for each robot:
       cell ← world→cell(TF pose), snapped to the nearest nav_free cell
       set env robot pos / cell / heading
       if it has a goal, end the goal when:
           Nav2 says succeeded                     → clear
           Nav2 says failed or rejected            → clear + blacklist the goal (30 s)
           the 1 m area around the goal is known   → clear (same rule as training)
           more than 90 s since it was sent        → clear + blacklist
7. log a CSV row: sim time, known m², decisions, failed goals
8. frontiers ← detect_frontiers(env.belief)
9. for each idle robot:
       (obs, mask, candidates) ← env._decision(robot, frontiers)    # training code
       action ← policy (PPO: model.predict(obs, mask, deterministic=True))
       send candidates[action] to that robot's Nav2 as a NavigateToPose goal
       write the goal into the env immediately, so the other robot's decision
       in the same tick sees it (as in training)
10. if no robot could be given a goal and both are idle: log EXPLORATION COMPLETE
```

### 13.4 Map resampling (`_resample_map()`)
The merged `/map` has its own origin, size and resolution. The env grid is a fixed 170 × 170 at 0.05 m centred on the arena. For every env cell centre, the node computes which `/map` cell contains it, reads its value, and converts it to the env's encoding: `< 0 → −1` (unknown), `≥ 50 → 100` (occupied), otherwise `0` (free). Env cells outside `/map` are unknown. The whole thing is a single vectorised NumPy lookup, with no loops.

### 13.5 Goals and Nav2
Goals are sent as `NavigateToPose` in the `map` frame, with an identity orientation (the robot's final heading doesn't matter for exploration). Each goal carries a unique **token**. Late callbacks from a goal that has since been replaced are ignored by comparing tokens. This prevents a classic race where an old goal's "succeeded" result clears a new goal.

`--policy heuristic` and `--policy nearest` run the **same node** with a different decision rule. In the Gazebo benchmark, both sides therefore have identical map handling, goal handling, timeouts and blacklisting. Only the frontier choice differs.

### 13.6 The sim-to-sim gap
| | 2D simulator | Gazebo stack |
|---|---|---|
| Mapping | ideal: everything a ray reaches is known instantly | SLAM integrates scans gradually; the map lags and has noise |
| Area known at start | ~36 m² | ~4 m² |
| Motion | A* path followed exactly at 0.15 m/s | Nav2 global planner + DWB controller: rotations, slow-downs, recoveries |
| Goal failures | impossible (reachability checked) | possible (Nav2 aborts) → blacklist |
| Time to map most of the arena | ~70–110 s | > 240 s |

The policy never saw SLAM lag or Nav2 failures during training. That it still helps in Gazebo shows that its decisions depend on features that keep their meaning across the gap: relative positions, distances, unknown fraction, the other robot's goal.

---

## 14. The Gazebo benchmark and its statistics

### 14.1 One run: `rl_sim/gazebo_test.sh <ppo|heuristic|nearest> [seconds] [model] [world]`
1. Isolates the run: `ROS_DOMAIN_ID=77`, its own Gazebo master port, no display.
2. Launches, with fixed waits between them: headless Gazebo on the benchmark arena (25 s), SLAM ×2 (10 s), map merge (8 s), Nav2 ×2 (35 s).
3. Runs `ros_policy_node.py --policy … --log results/gazebo_runs/<policy>_<time>.csv --max-time 240`.
4. Kills every process group on exit, even after an error.

The CSV has one row per second of sim time: `sim_time_s, known_m2, decisions, failed_goals`.

### 14.2 Experimental design
- **Alternating pairs:** PPO, heuristic, PPO, heuristic, … The k-th PPO run is paired with the k-th heuristic run. Anything that drifts over time (machine temperature, background load) affects both members of a pair about equally, and the paired difference cancels it.
- **Why there is variance at all:** Gazebo physics, SLAM and Nav2 aren't deterministic. Thread timing changes results run to run, even with the same policy on the same arena.

### 14.3 Metrics (`analyze_gazebo_tests.py`)
- known area at 60 / 120 / 180 / 240 s;
- mean known area over 0–240 s (area under the curve ÷ 240);
- **time to reach 30 m²** (the primary speed metric);
- failed goals and decisions per run.

### 14.4 The exact paired permutation test
For each metric: compute the 8 paired differences $d_k = \text{PPO}_k - \text{Heuristic}_k$. Under the null hypothesis "the policy makes no difference", each difference is equally likely to be positive or negative. The test therefore enumerates **all $2^8 = 256$ sign flips** of the differences and asks: in what fraction is $|\text{mean}|$ at least as large as the observed one?

$$p = \frac{\#\{\text{sign patterns } s : |\overline{s \odot d}| \ge |\bar{d}|\}}{256}$$

The test is exact (not an approximation) and assumes nothing about normality, which matters with only 8 samples. The smallest possible two-sided p is 2/256 ≈ **0.0078**, reached when all 8 differences have the same sign.

---

## 15. Results and what the policy learned

From [heuristic-vs-rl.md](heuristic-vs-rl.md) (8 paired Gazebo runs of 240 s on the benchmark arena):

| | PPO | Heuristic | Difference |
|---|---|---|---|
| **Time to map 30 m²** | **127 ± 15 s** | 160 ± 30 s | −33 s (**−21 %**), PPO faster in **8/8** pairs, p = 0.008 |
| Known area at 180 s | **36.7 ± 2.8 m²** | 32.4 ± 3.8 m² | +13 %, 8/8 pairs, p = 0.008 |
| Known area at 240 s | 41.3 ± 2.3 m² | 38.4 ± 2.2 m² | +7 %, 6/8 pairs, p = 0.055 (not significant) |
| 2D simulator, time to explore | **67 s** | 111 s | −40 % (deterministic) |

> **Note on configuration.** These Gazebo results were measured with the Nav2 configuration that was current at the time ([navigation-fix.md](navigation-fix.md)). The Nav2 parameters have since changed (global planner switched to NavFn, stronger obstacle critic), and the newer runs in `results/gazebo_runs/` are not yet a full 8-pair set. Re-run the benchmark before quoting numbers for the current configuration.

**What the policy does differently:**
- **It weighs information gain.** In the traced first decision ([5.4](#54-a-real-observation)) it passes over the nearest frontier, which has little unknown around it, for an equally near one with three times as much.
- **It makes more, shorter trips in Gazebo** (21 vs 17 decisions in 240 s). The heuristic's large penalties (50× for nearness to the other robot's goal, 50× per metre into the other half) push it towards distant frontiers. The policy takes nearer ones and redirects as soon as an area is mapped.
- **The advantage is largest mid-run.** In the first minute both policies are mapping around their start points. Near the end only small, scattered frontiers remain and the choice matters less. In between, good choices compound.

---

## 16. Design decisions and why they were made

### 16.1 Decide frontiers, not velocities
An agent that outputs wheel velocities would need thousands of steps per episode, would have to learn obstacle avoidance and path following from scratch, and would duplicate what Nav2 already does well. Deciding *which frontier* puts learning exactly where the hand-designed system is weakest, keeps episodes to ~7–10 steps, and makes deployment trivial: the output is a Nav2 goal.

### 16.2 Stable slot semantics
Actions are slots in a list sorted by distance, so "slot 0" always means "nearest valid frontier". The alternative, an action per map cell, gives a huge action space in which most actions are invalid, and the meaning of an action shifts as the map changes. The docstring of `frontier_env.py` records that an earlier per-cell formulation did not converge. Slots plus masking fixed this.

### 16.3 A feature vector, not a map image
A CNN over a 170 × 170 map image (28,900 cells) would have to *discover* concepts like "distance to the other robot's goal" from pixels, needing far more data and a larger network. Hand-designed, ego-centric features hand it those concepts directly. The trade-off: the policy can only use what the features expose. It can't notice, say, a corridor shape that no feature describes.

### 16.4 A fast, deterministic simulator
Gazebo episodes take minutes. The 2D simulator makes 300k decisions (~40k episodes) possible in 25 minutes on a laptop. Determinism gives a clean learning signal: the same choices always give the same result, so differences in return are entirely due to the policy, and a single evaluation episode is exact.

### 16.5 Matching the real stack wherever it matters
Same world file, same resolution, same frontier detector, same lidar range, same speed, same clearance (0.30 m inflation ≈ Nav2's 0.35 m), same goal-ending rule, same candidate filtering, and the same observation code at deployment. Every mismatch is a potential sim-to-sim failure. The remaining differences (ideal sensing, ideal motion) are deliberate simplifications.

### 16.6 Reward = new area − time
Simple, dense and well-scaled, and it optimises exactly the target metric ([3.7](#37-reward)). There are no hand-tuned bonus terms whose weights could encode unintended behaviour.

### 16.7 What was tried before
- **A per-cell action space** (pick any map cell as a goal): did not converge ([16.2](#162-stable-slot-semantics)).
- **Training from scratch with a saturating reward and many unreachable goals** learned nothing. The current design fixes both: reachability is checked through connected components, and reward is per m² and per second.
- **Snapping goals onto the frontier edge** (next to unknown space) halved exploration in Gazebo: goals sat too close to unknown and obstacles. Goals now go to the nearest *reachable, inflated-free* cell near the centroid instead.
- **Inflation 0.15 m:** robots got pinned against obstacles in Gazebo. The simulator's clearance was raised to 0.30 m to match the stronger Nav2 inflation, and the policy was retrained.

---

## 17. Limitations

1. **Arena.** Training and evaluation use the benchmark arena. Performance on other layouts was not measured, so no claim is made about them.
2. **Ideal 2D world.** No sensor noise, no SLAM lag, no navigation failures in training. The policy has never had to cope with these. It works in Gazebo despite them, not because it learned to handle them.
3. **Fixed horizon in Gazebo.** Runs last 240 s, and neither policy fully explores the arena in that time in Gazebo, so "time to full exploration" was only measured in 2D.
4. **Sample size.** 8 pairs. The primary metrics reach the smallest achievable p (0.008), but metrics around p ≈ 0.02 wouldn't all survive a strict multiple-comparison correction.
5. **Two robots, simulation only.** The observation layout assumes exactly one "other" robot. No physical robots were used.
6. **The benchmark's heuristic is the decision rule, not the full coordinator node.** Both policies ran inside `ros_policy_node.py`, with identical goal handling. The original `frontier_coordinator` also does stuck detection, collision pausing and 3 s replanning. This isolates the decision rule, which is the claim being made.

---

## 18. How to change things

### 18.1 Add an observation feature
1. In `frontier_env.py`, increase `CANDIDATE_FEATURES` (or `GLOBAL_FEATURES`). `OBS_DIM` updates automatically.
2. Append the value in `_decision()` to the `feats[k] = [...]` list (or `glob`), scaled to roughly [−1, 1].
3. Retrain. The old model expects 153 inputs and will refuse to load against the new space.
4. Nothing to change in `ros_policy_node.py`: it calls the same `_decision()`.

### 18.2 Change the reward
Edit `AREA_REWARD` / `TIME_PENALTY`, or the reward line in `step()`. Keep per-step rewards of order 0.1–1. If you add terms, check what return a finished episode gets, as in [3.7](#37-reward), so you know what's actually being optimised.

### 18.3 Change the number of candidates
`MAX_CANDIDATES` sets both the observation size and `Discrete(n)`. It's rare for more than 8 to be valid on this arena.

### 18.4 Change the simulator to match the real stack
`SPEED`, `DT`, `ROBOT_INFLATION` (world.py), `LIDAR_RANGE` (lidar.py), and the frontier parameters (frontiers.py, kept in sync with `frontier_coordinator.py`). Change both sides together.

### 18.5 Retrain and redeploy
```bash
python3 rl_sim/train.py                       # → ~/rl_sim_runs/ppo_<time>/best_model.zip
python3 rl_sim/evaluate.py --model ~/rl_sim_runs/ppo_<time>/best_model.zip
cp ~/rl_sim_runs/ppo_<time>/best_model.zip models/ppo_frontier_policy.zip
cp ~/rl_sim_runs/ppo_<time>/{eval_history,tensorboard_scalars}.csv results/training/
python3 rl_sim/make_graphs.py
```
Then repeat the Gazebo benchmark ([14](#14-the-gazebo-benchmark-and-its-statistics)) before claiming anything about the real stack.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **Action mask** | Boolean array marking which actions are valid. Invalid ones get probability 0. |
| **Actor** | The policy network $\pi_\theta(a \mid o)$. |
| **Advantage** $A(s, a)$ | How much better an action was than the policy's average in that state: $Q - V$. |
| **Benchmark arena** | Arena number 42 of `generate_random_world`, used for training and all evaluations. |
| **Candidate** | A frontier, snapped to a reachable goal cell, that passed all filters. Up to 12 per decision. |
| **Clip range** $\epsilon$ | PPO's limit on how much one update can change an action's probability (0.2). |
| **Critic** | The value network $V_\phi(o)$. |
| **Entropy** | Spread of the policy's distribution. High means exploring, ~0 means deterministic. |
| **Episode** | One exploration from reset until nothing is left to explore (~7–10 decisions). |
| **Explained variance** | How much of the variation in returns the critic predicts (1 = perfectly). |
| **Frontier** | Boundary between known free cells and unknown cells. |
| **GAE** | Generalised Advantage Estimation: advantage estimates blending TD and Monte-Carlo with $\lambda$. |
| **Heuristic** | `frontier_coordinator`'s hand-designed cost formula. |
| **KL divergence** | How different two probability distributions are; here, the policy before vs after an update. |
| **Logit** | Unnormalised score; softmax turns logits into probabilities. |
| **MaskablePPO** | PPO with action masking (`sb3-contrib`). |
| **MDP** | Markov Decision Process: states, actions, transitions, rewards, discount. |
| **MLP** | Multi-layer perceptron: stacked fully connected layers. |
| **Observation** | The 153 numbers the policy sees. |
| **On-policy** | Learns only from data collected by the current policy. |
| **Option / semi-MDP** | An action that lasts a variable amount of time (here: driving to a frontier). |
| **Parameter sharing** | Both robots use the same network. |
| **Policy** | Mapping from observation to (a distribution over) actions. |
| **PPO** | Proximal Policy Optimization: actor-critic policy gradient with a clipped update. |
| **Return** | Discounted sum of future rewards. |
| **Rollout** | A batch of experience collected with the current policy (2,048 transitions here). |
| **Sim-to-sim transfer** | Running a policy trained in one simulator in a different one without retraining. |
| **Slot** | Position in the distance-sorted candidate list. The action is a slot index. |
| **TD error** | One-step estimate of advantage: $r + \gamma V(s') - V(s)$. |
| **Zero-shot** | Deployed without any further training. |

---

## 20. Command reference

Run inside the ROS 2 environment (`distrobox enter ros-humble`) from `final-product/`.

```bash
# one-time setup
pip3 install --user torch --index-url https://download.pytorch.org/whl/cpu
pip3 install --user -r rl_sim/requirements.txt      # gymnasium, stable-baselines3, sb3-contrib, tensorboard, numpy, opencv
pip3 uninstall -y setuptools                         # torch's setuptools breaks colcon on Humble
colcon build --symlink-install

# train (~25 min) and watch it
python3 rl_sim/train.py
tensorboard --logdir ~/rl_sim_runs

# compare policies in 2D (add --video for MP4s)
python3 rl_sim/evaluate.py --model models/ppo_frontier_policy.zip

# one headless Gazebo run (~5 min)
rl_sim/gazebo_test.sh ppo 240 models/ppo_frontier_policy.zip
rl_sim/gazebo_test.sh heuristic 240

# statistics and charts
python3 rl_sim/analyze_gazebo_tests.py
python3 rl_sim/count_nav2_failures.py results/gazebo_runs
python3 rl_sim/make_graphs.py

# run the policy live with the normal 5-terminal setup (instead of terminal 5)
GAZEBO_WORLD_SEED=42 ros2 launch multi_robot_exploration spawn_two_turtlebots.launch.py   # terminal 1
python3 rl_sim/ros_policy_node.py --model models/ppo_frontier_policy.zip                    # terminal 5
```

| File | Role |
|---|---|
| `rl_sim/core/world.py` | arena from the Gazebo world file; inflation; connected components |
| `rl_sim/core/lidar.py` | vectorised 360° ray-caster |
| `rl_sim/core/frontiers.py` | frontier detection identical to `frontier_coordinator.py` |
| `rl_sim/core/planner.py` | 8-connected A* |
| `rl_sim/envs/frontier_env.py` | the Gymnasium environment, observation, mask, reward, baselines |
| `rl_sim/train.py` | MaskablePPO training, evaluation callback, best-model saving |
| `rl_sim/evaluate.py` | 2D comparison of PPO and baselines |
| `rl_sim/ros_policy_node.py` | runs a policy on the ROS 2 / Gazebo stack |
| `rl_sim/gazebo_test.sh` | one headless benchmark run |
| `rl_sim/analyze_gazebo_tests.py` | paired statistics |
| `rl_sim/make_graphs.py` | every chart in `graphs/` |
| `rl_sim/export_tensorboard.py` | TensorBoard → CSV |
| `models/ppo_frontier_policy.zip` | the shipped policy (30,008-decision checkpoint) |
| `results/training/` | evaluation history and all training scalars |
| `results/sim2d/` | 2D comparison |
| `results/gazebo_runs/` | per-second known area for each Gazebo run |
