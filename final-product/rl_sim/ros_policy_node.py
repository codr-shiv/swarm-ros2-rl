#!/usr/bin/env python3
"""
Run the rl_sim policy on the real Gazebo/SLAM/Nav2 stack (sim-to-sim transfer).

Replaces terminal 5 (frontier_exploration.launch.py), on the benchmark arena:

  GAZEBO_WORLD_SEED=42 ros2 launch multi_robot_exploration spawn_two_turtlebots.launch.py
  ... SLAM, map merge, Nav2 as usual ...
  python3 rl_sim/ros_policy_node.py --model models/ppo_frontier_policy.zip

How the observation is kept identical to training: the node holds a
FrontierExplorationEnv for the same world and overwrites its state every
second with the real one: the merged /map resampled onto the env's 0.05 m
grid, robot poses and headings from TF, current goals, elapsed sim time,
and the blacklist. The env's own _decision() then builds the observation and
action mask, and the chosen candidate is sent to that robot's Nav2.

--policy heuristic|nearest runs the same node with a baseline instead of PPO,
so comparisons in Gazebo use exactly the same goal handling.
"""
import argparse
import csv
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from action_msgs.msg import GoalStatus  # noqa: E402
from nav2_msgs.action import NavigateToPose  # noqa: E402
from nav_msgs.msg import OccupancyGrid  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy  # noqa: E402
from rclpy.signals import SignalHandlerOptions  # noqa: E402
from tf2_ros import Buffer, TransformListener  # noqa: E402

from rl_sim.core.frontiers import detect_frontiers  # noqa: E402
from rl_sim.core.world import DEFAULT_WORLD  # noqa: E402
from rl_sim.envs.frontier_env import FrontierExplorationEnv  # noqa: E402

ROBOTS = ('robot1', 'robot2')
GOAL_TIMEOUT_S = 90.0       # sim s; Nav2 normally finishes or aborts much sooner
TICK_S = 1.0


class PolicyCoordinator(Node):

    def __init__(self, args):
        super().__init__('rl_policy_coordinator')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.args = args
        self.env = FrontierExplorationEnv(args.world)
        self.env.blacklist = {}
        self.env.t = 0.0
        w = self.env.world
        # world (x, y) of every env cell centre, for resampling /map
        idx = np.arange(w.grid.shape[0])
        self.cell_y = (w.origin + (idx + 0.5) * w.resolution)[:, None]
        self.cell_x = (w.origin + (idx + 0.5) * w.resolution)[None, :]

        if args.policy == 'ppo':
            from sb3_contrib import MaskablePPO
            self.model = MaskablePPO.load(os.path.expanduser(args.model))
        self._lock = threading.Lock()
        self.latest_map = None
        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, qos)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = {r: ActionClient(self, NavigateToPose, f'/{r}/navigate_to_pose') for r in ROBOTS}

        self.state = {r: {'goal': None, 'handle': None, 'sent': None, 'done': None, 'token': None}
                      for r in ROBOTS}
        self.start = None
        self.finished_at = None
        self.decisions = 0
        self.failures = 0
        self.csv = None
        if args.log:
            os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
            self.csv = open(args.log, 'w', newline='')
            self.writer = csv.writer(self.csv)
            self.writer.writerow(['sim_time_s', 'known_m2', 'decisions', 'failed_goals'])
        self.create_timer(TICK_S, self._tick)
        self.get_logger().info(f'policy={args.policy}; waiting for Nav2/map/TF')

    # ── inputs ───────────────────────────────────────────────────────────
    def _map_cb(self, msg):
        self.latest_map = msg

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _pose(self, robot):
        try:
            t = self.tf_buffer.lookup_transform('map', f'{robot}/base_footprint', rclpy.time.Time())
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def _resample_map(self, msg):
        """Merged /map -> env grid (-1 unknown, 0 free, 100 occupied)."""
        info = msg.info
        data = np.array(msg.data, dtype=np.int16).reshape(info.height, info.width)
        col = np.floor((self.cell_x - info.origin.position.x) / info.resolution).astype(int)
        row = np.floor((self.cell_y - info.origin.position.y) / info.resolution).astype(int)
        inside = (row >= 0) & (row < info.height) & (col >= 0) & (col < info.width)
        rr = np.clip(row, 0, info.height - 1)
        cc = np.clip(col, 0, info.width - 1)
        vals = data[np.broadcast_to(rr, inside.shape), np.broadcast_to(cc, inside.shape)]
        belief = np.where(vals < 0, -1, np.where(vals >= 50, 100, 0)).astype(np.int16)
        belief[~inside] = -1
        return belief

    def _snap(self, row, col):
        """Nearest cell the env considers drivable (inflated map) to (row, col)."""
        nav = self.env.world.nav_free
        h, w = nav.shape
        row, col = min(max(row, 0), h - 1), min(max(col, 0), w - 1)
        if nav[row, col]:
            return row, col
        for rad in range(1, 15):
            win = nav[max(row - rad, 0):row + rad + 1, max(col - rad, 0):col + rad + 1]
            rr, cc = np.nonzero(win)
            if len(rr):
                rr = rr + max(row - rad, 0)
                cc = cc + max(col - rad, 0)
                k = int(np.argmin((rr - row) ** 2 + (cc - col) ** 2))
                return int(rr[k]), int(cc[k])
        return row, col

    # ── Nav2 goals ───────────────────────────────────────────────────────
    def _send(self, robot, cell):
        x, y = self.env.world.to_world(*cell)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.w = 1.0
        token = object()
        with self._lock:
            st = self.state[robot]
            st.update(goal=cell, handle=None, sent=self._now(), done=None, token=token)
        self.nav[robot].send_goal_async(goal).add_done_callback(
            lambda f: self._on_response(f, robot, token))

    def _on_response(self, fut, robot, token):
        try:
            handle = fut.result()
        except Exception:
            handle = None
        with self._lock:
            st = self.state[robot]
            if st['token'] is not token:
                return
            if handle is None or not handle.accepted:
                st['done'] = 'rejected'
                return
            st['handle'] = handle
        handle.get_result_async().add_done_callback(lambda f: self._on_result(f, robot, token))

    def _on_result(self, fut, robot, token):
        try:
            status = fut.result().status
        except Exception:
            status = GoalStatus.STATUS_ABORTED
        with self._lock:
            st = self.state[robot]
            if st['token'] is token:
                st['done'] = 'succeeded' if status == GoalStatus.STATUS_SUCCEEDED else 'failed'

    def _clear(self, robot, blacklist=False):
        with self._lock:
            st = self.state[robot]
            if blacklist and st['goal'] is not None:
                self.env.blacklist[st['goal']] = self.env.t
                self.failures += 1
            if st['handle'] is not None:
                try:
                    st['handle'].cancel_goal_async()
                except Exception:
                    pass
            st.update(goal=None, handle=None, sent=None, done=None, token=None)

    # ── main loop ────────────────────────────────────────────────────────
    def _tick(self):
        if not all(c.server_is_ready() for c in self.nav.values()) or self.latest_map is None:
            return
        poses = {r: self._pose(r) for r in ROBOTS}
        if any(p is None for p in poses.values()):
            return
        now = self._now()
        if self.start is None:
            self.start = now
            for i, r in enumerate(ROBOTS):          # home = start position (region split)
                row, col = self.env.world.to_cell(poses[r][0], poses[r][1])
                self.env_robot(i)['home'] = np.array(self._snap(row, col), float)
            self.get_logger().info('Exploration started.')
        env = self.env
        env.t = now - self.start
        env.belief = self._resample_map(self.latest_map)
        env.blacklist = {g: t for g, t in env.blacklist.items() if env.t - t < 30.0}

        # Sync robot states into the env
        for i, r in enumerate(ROBOTS):
            x, y, yaw = poses[r]
            row, col = env.world.to_cell(x, y)
            cell = self._snap(row, col)
            er = self.env_robot(i)
            er['pos'] = np.array(cell, float)
            er['cell'] = cell
            er['heading'] = yaw
            st = self.state[r]
            if st['goal'] is not None:
                if st['done'] == 'succeeded':
                    self._clear(r)
                elif st['done'] in ('failed', 'rejected'):
                    self._clear(r, blacklist=True)
                elif env._goal_explored(st['goal']):          # same rule as in training
                    self._clear(r)
                elif now - st['sent'] > GOAL_TIMEOUT_S:
                    self._clear(r, blacklist=True)
            er['goal'] = np.array(self.state[r]['goal'], float) if self.state[r]['goal'] else None

        if self.csv:
            self.writer.writerow([round(env.t, 1), round(env._known_m2(), 2), self.decisions, self.failures])
            self.csv.flush()

        # Decide for idle robots
        frontiers, sizes = detect_frontiers(env.belief, env.world.resolution)
        idle = [i for i, r in enumerate(ROBOTS) if self.state[r]['goal'] is None]
        decided = False
        for i in idle:
            d = env._decision(i, frontiers, sizes)
            if d is None:
                continue
            obs, mask, cands = d
            env.pending = (i, obs, mask, cands)
            if self.args.policy == 'ppo':
                a = int(self.model.predict(obs, action_masks=mask, deterministic=True)[0])
            elif self.args.policy == 'heuristic':
                a = env.heuristic_action()
            else:
                a = env.nearest_action()
            if not mask[a]:
                a = int(np.flatnonzero(mask)[0])
            self._send(ROBOTS[i], cands[a])
            self.env_robot(i)['goal'] = np.array(cands[a], float)   # visible to the other robot's decision
            self.decisions += 1
            decided = True
            self.get_logger().info(f'{ROBOTS[i]} -> {env.world.to_world(*cands[a])} (slot {a})')
        if not decided and len(idle) == len(ROBOTS) and self.finished_at is None:
            self.finished_at = env.t
            self.get_logger().info(f'EXPLORATION COMPLETE at {env.t:.1f} s sim time: '
                                   f'{env._known_m2():.1f} m2 known, {self.decisions} decisions, '
                                   f'{self.failures} failed goals')

    def env_robot(self, i):
        if not hasattr(self.env, 'robots'):
            self.env.robots = [{'pos': None, 'cell': None, 'goal': None, 'path': [],
                                'heading': 0.0, 'home': None} for _ in ROBOTS]
        return self.env.robots[i]

    def close(self):
        for r in ROBOTS:
            self._clear(r)
        if self.csv:
            self.csv.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', help='MaskablePPO .zip (required for --policy ppo)')
    p.add_argument('--world', type=int, default=DEFAULT_WORLD,
                   help='arena number; must equal GAZEBO_WORLD_SEED of the running Gazebo')
    p.add_argument('--policy', choices=['ppo', 'heuristic', 'nearest'], default='ppo')
    p.add_argument('--log', help='CSV of known area over sim time')
    p.add_argument('--max-time', type=float, default=0, help='exit after this many sim s (0 = never)')
    args = p.parse_args()
    if args.policy == 'ppo' and not args.model:
        p.error('--model is required for --policy ppo')

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = PolicyCoordinator(args)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.max_time and node.start is not None and node.env.t >= args.max_time:
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
        time.sleep(0.2)


if __name__ == '__main__':
    main()
