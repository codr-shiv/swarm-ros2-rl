"""
Two-robot frontier exploration, decision-level, on the Gazebo arena.

One step = one decision: which frontier the robot that needs a goal should
explore next. The robot then follows an A* path at TurtleBot3 speed while
both lidars update the shared map, until some robot needs a new goal. This
mirrors the ROS stack (frontier_coordinator picks a goal, Nav2 drives).

Why this converges (unlike the per-cell version):
  - actions are stable: slot k = the k-th nearest reachable frontier, and
    invalid slots are masked (MaskablePPO), so every action is meaningful;
  - one action = one whole trip, so episodes are ~10 decisions, not hundreds;
  - the observation is a compact feature vector including positions;
    the simulator is deterministic (no noise);
  - reward is small and well scaled: +0.1 per new m², -0.01 per second.
"""
import math

import cv2  # type: ignore[import-untyped]
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_sim.core.frontiers import detect_frontiers
from rl_sim.core.lidar import Lidar
from rl_sim.core.planner import astar
from rl_sim.core.world import DEFAULT_WORLD, FREE, load_world

MAX_CANDIDATES = 12
CANDIDATE_FEATURES = 12
GLOBAL_FEATURES = 9
OBS_DIM = MAX_CANDIDATES * CANDIDATE_FEATURES + GLOBAL_FEATURES

SPEED = 0.15                # m/s, TurtleBot3 Burger max (Nav2 config)
DT = 0.5                    # s per simulation tick
MAX_EPISODE_S = 600.0
GOAL_SNAP_RADIUS = 0.6      # m: frontier centroid -> nearest reachable cell
MIN_GOAL_DISTANCE = 0.6     # m (coordinator)
OTHER_GOAL_EXCLUSION = 1.5  # m (coordinator dedup radius)
GAIN_RADIUS = 1.0           # m, window for the unknown-fraction feature
ARENA = 4.0                 # m, half arena (feature scaling)

MAX_DECISIONS = 400         # safety cap per episode (normal: 30-80)
BLACKLIST_S = 30.0          # unreachable goals are skipped this long

AREA_REWARD = 0.1           # per m² newly known
TIME_PENALTY = 0.01         # per simulated second

# frontier_coordinator.py's cost, used as a feature and as a baseline policy
HEUR_SEPARATION, HEUR_REGION = 50.0, 50.0


class FrontierExplorationEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, world=DEFAULT_WORLD):
        super().__init__()
        self.world = load_world(world)
        self.lidar = Lidar(self.world.resolution)
        self.observation_space = spaces.Box(-5.0, 5.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(MAX_CANDIDATES)
        w = self.world
        self.total_free_m2 = float(np.count_nonzero(w.grid == FREE)) * w.resolution ** 2
        self.gain_k = 2 * int(round(GAIN_RADIUS / w.resolution)) + 1
        self.pending = None
        self.on_tick = None        # optional callback(env) after every simulated tick (video recording)

    # ── gym API ──────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        w = self.world
        self.belief = np.full(w.grid.shape, -1, dtype=np.int16)
        self.robots = []
        for row, col in w.spawns:
            self.robots.append({'pos': np.array([row, col], float), 'cell': (row, col),
                                'goal': None, 'path': [], 'heading': 0.0,
                                'home': np.array([row, col], float)})
        for r in self.robots:
            self._scan(r)
        self.t = 0.0
        self.decisions = 0
        self.blacklist = {}            # goal cell -> sim time it was found unreachable
        self.prev_known = self._known_m2()
        self.turn = 0
        end = self._advance()
        if end is not None:
            raise RuntimeError('nothing to explore at reset')
        return self.pending[1], self._info()

    def step(self, action):
        idx, obs, mask, cands = self.pending
        a = int(action)
        if not mask[a]:
            a = int(np.flatnonzero(mask)[0])
        robot = self.robots[idx]
        path = astar(self.world.nav_free, robot['cell'], cands[a])
        if path and len(path) > 1:
            robot['goal'] = np.array(cands[a], float)
            robot['path'] = path[1:]
        else:
            self.blacklist[cands[a]] = self.t       # shouldn't happen (same component)
        self.decisions += 1

        t0 = self.t
        end = self._advance()
        if end is None and self.decisions >= MAX_DECISIONS:
            end, self.pending = 'time_limit', None
        known = self._known_m2()
        reward = AREA_REWARD * (known - self.prev_known) - TIME_PENALTY * (self.t - t0)
        self.prev_known = known
        terminated = end == 'explored'
        truncated = end == 'time_limit'
        obs = self.pending[1] if end is None else np.zeros(OBS_DIM, np.float32)
        return obs, float(reward), terminated, truncated, self._info(end)

    def action_masks(self):
        return self.pending[2] if self.pending is not None else np.ones(MAX_CANDIDATES, bool)

    def heuristic_action(self):
        """frontier_coordinator's choice among the current candidates."""
        idx, obs, mask, cands = self.pending
        costs = [self._heuristic_cost(idx, c) if mask[k] else np.inf for k, c in enumerate(cands)]
        costs += [np.inf] * (MAX_CANDIDATES - len(costs))
        return int(np.argmin(costs))

    def nearest_action(self):
        return int(np.flatnonzero(self.pending[2])[0])   # candidates are sorted by distance

    def render(self):
        img = np.full(self.belief.shape + (3,), 110, np.uint8)
        img[self.belief == FREE] = (255, 255, 255)
        img[self.belief > 50] = (0, 0, 0)
        for r, color in zip(self.robots, ((255, 160, 0), (0, 120, 255))):
            p = (int(r['pos'][1]), int(r['pos'][0]))
            cv2.circle(img, p, 3, color, -1)
            if r['goal'] is not None:
                cv2.drawMarker(img, (int(r['goal'][1]), int(r['goal'][0])), color, cv2.MARKER_CROSS, 8, 2)
        return cv2.flip(img, 0)   # row 0 = -y at the bottom

    # ── simulation ───────────────────────────────────────────────────────
    def _scan(self, robot):
        r, c = robot['cell']
        self.lidar.scan(self.belief, self.world.grid, r, c)

    def _known_m2(self):
        return float(np.count_nonzero(self.belief >= 0)) * self.world.resolution ** 2

    def _goal_explored(self, goal):
        """True once the area around the goal is fully known (frontier 'vanished')."""
        r, c = int(goal[0]), int(goal[1])
        k = self.gain_k // 2
        win = self.belief[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1]
        return not (win == -1).any()

    def _move(self, robot):
        """Advance one tick along the path; returns True when the robot became idle."""
        step_cells = SPEED * DT / self.world.resolution
        while step_cells > 0 and robot['path']:
            nxt = np.array(robot['path'][0], float)
            d = np.linalg.norm(nxt - robot['pos'])
            if d <= step_cells:
                robot['heading'] = math.atan2(nxt[0] - robot['pos'][0], nxt[1] - robot['pos'][1]) \
                    if d > 0 else robot['heading']
                robot['pos'] = nxt
                robot['cell'] = robot['path'].pop(0)
                step_cells -= d
            else:
                robot['heading'] = math.atan2(nxt[0] - robot['pos'][0], nxt[1] - robot['pos'][1])
                robot['pos'] = robot['pos'] + (nxt - robot['pos']) * (step_cells / d)
                step_cells = 0
        self._scan(robot)
        if not robot['path'] or self._goal_explored(robot['goal']):
            robot['goal'], robot['path'] = None, []
            return True
        return False

    def _advance(self):
        """Simulate until a robot needs a decision (sets self.pending) or the episode ends."""
        while True:
            if self.t >= MAX_EPISODE_S:
                self.pending = None
                return 'time_limit'
            idle = [i for i, r in enumerate(self.robots) if r['goal'] is None]
            if idle:
                frontiers, sizes = detect_frontiers(self.belief, self.world.resolution)
                order = idle if self.turn % 2 == 0 else idle[::-1]
                for i in order:
                    d = self._decision(i, frontiers, sizes)
                    if d is not None:
                        self.turn += 1
                        self.pending = (i,) + d
                        return None
                if len(idle) == len(self.robots):
                    self.pending = None
                    return 'explored'          # nobody can go anywhere useful
            for r in self.robots:
                if r['goal'] is not None:
                    self._move(r)
            self.t += DT
            if self.on_tick is not None:
                self.on_tick(self)

    # ── observation ──────────────────────────────────────────────────────
    def _candidates(self, i, frontiers, sizes):
        """Snapped, reachable, allowed goals for robot i, nearest first."""
        w = self.world
        ego, other = self.robots[i], self.robots[1 - i]
        comp = w.component[ego['cell']]
        rad = int(GOAL_SNAP_RADIUS / w.resolution)
        self.blacklist = {g: t for g, t in self.blacklist.items() if self.t - t < BLACKLIST_S}
        out = []
        for f, s in zip(frontiers, sizes):
            r0, c0 = int(f[0]), int(f[1])
            rows = slice(max(r0 - rad, 0), r0 + rad + 1)
            cols = slice(max(c0 - rad, 0), c0 + rad + 1)
            ok = w.nav_free[rows, cols] & (w.component[rows, cols] == comp)
            rr, cc = np.nonzero(ok)
            if len(rr) == 0:
                continue                                   # unreachable frontier
            rr, cc = rr + rows.start, cc + cols.start
            k = int(np.argmin((rr - f[0]) ** 2 + (cc - f[1]) ** 2))
            goal = (int(rr[k]), int(cc[k]))
            if goal in self.blacklist or self._goal_explored(goal):
                continue                                   # nothing left to see there
            dist = np.linalg.norm(np.array(goal) - ego['pos']) * w.resolution
            if dist < MIN_GOAL_DISTANCE:
                continue
            if other['goal'] is not None and \
                    np.linalg.norm(np.array(goal) - other['goal']) * w.resolution < OTHER_GOAL_EXCLUSION:
                continue
            out.append((dist, goal, s))
        out.sort(key=lambda t: t[0])
        return out[:MAX_CANDIDATES]

    def _heuristic_cost(self, i, goal):
        res = self.world.resolution
        ego, other = self.robots[i], self.robots[1 - i]
        g = np.array(goal, float)
        cost = np.linalg.norm(g - ego['pos']) * res
        if other['goal'] is not None:
            cost += HEUR_SEPARATION / max(np.linalg.norm(g - other['goal']) * res, 0.1)
        axis = (other['home'] - ego['home']) * res
        n = np.linalg.norm(axis)
        if n > 0.1:
            proj = np.dot((g - (ego['home'] + other['home']) / 2) * res, axis / n)
            cost += max(0.0, proj) * HEUR_REGION
        return cost

    def _decision(self, i, frontiers, sizes):
        cands = self._candidates(i, frontiers, sizes)
        if not cands:
            return None
        w = self.world
        res = w.resolution
        ego, other = self.robots[i], self.robots[1 - i]
        gain = cv2.boxFilter((self.belief == -1).astype(np.float32), -1, (self.gain_k, self.gain_k),
                             borderType=cv2.BORDER_CONSTANT)
        heur = [self._heuristic_cost(i, g) for _, g, _ in cands]
        best = int(np.argmin(heur))

        def xy(rc):   # cell -> metres, scaled to about [-1, 1]
            x, y = w.to_world(rc[0], rc[1])
            return x / ARENA, y / ARENA

        feats = np.zeros((MAX_CANDIDATES, CANDIDATE_FEATURES), np.float32)
        for k, (dist, goal, size) in enumerate(cands):
            g = np.array(goal, float)
            gx, gy = xy(goal)
            ex, ey = xy(ego['pos'])
            bearing = math.atan2(g[0] - ego['pos'][0], g[1] - ego['pos'][1]) - ego['heading']
            d_other_goal = (np.linalg.norm(g - other['goal']) * res / 8.0) if other['goal'] is not None else 1.5
            feats[k] = [1.0, gx, gy, gx - ex, gy - ey, dist / 8.0,
                        np.linalg.norm(g - other['pos']) * res / 8.0, d_other_goal,
                        min(size / 100.0, 3.0), gain[goal], math.cos(bearing),
                        1.0 if k == best else 0.0]
        ex, ey = xy(ego['pos'])
        ox, oy = xy(other['pos'])
        ogx, ogy = xy(other['goal']) if other['goal'] is not None else (0.0, 0.0)
        glob = np.array([ex, ey, ox, oy, float(other['goal'] is not None), ogx, ogy,
                         self._known_m2() / max(self.total_free_m2, 1.0), self.t / MAX_EPISODE_S],
                        np.float32)
        obs = np.concatenate([feats.ravel(), glob])
        mask = np.zeros(MAX_CANDIDATES, bool)
        mask[:len(cands)] = True
        return obs, mask, [g for _, g, _ in cands]

    def _info(self, end=None):
        return {'explored_m2': self._known_m2(), 'sim_time_s': self.t,
                'decisions': self.decisions, 'end_reason': end}
