"""
Architecture diagrams (SVG), from system level down to function level.

  python3 docs/architecture/make_diagrams.py      # writes docs/architecture/*.svg

Edit the node/edge lists below to change a diagram.
"""
import os
from html import escape

OUT = os.path.dirname(os.path.abspath(__file__))
COL = {'sim': '#E8F1FB', 'ros': '#EAF6EC', 'nav': '#FFF4E0', 'rl': '#F3E8FA', 'io': '#F2F2F2', 'dec': '#FFE9E9'}
EDGE = {'sim': '#1F5FA8', 'ros': '#2E7D32', 'nav': '#B26A00', 'rl': '#7B3FA6', 'io': '#555555', 'dec': '#C62828'}


def diagram(name, title, w, h, nodes, edges, notes=()):
    """nodes: id -> (x, y, w, h, label, kind); edges: (src, dst, label, src_side, dst_side)."""
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="13">',
         '<defs><marker id="arr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
         '<path d="M0,0 L10,4 L0,8 z" fill="#333"/></marker></defs>',
         f'<rect width="{w}" height="{h}" fill="white"/>',
         f'<text x="{w/2}" y="30" text-anchor="middle" font-size="20" font-weight="bold">{title}</text>']

    def anchor(n, side):
        x, y, bw, bh = nodes[n][:4]
        return {'l': (x, y + bh / 2), 'r': (x + bw, y + bh / 2),
                't': (x + bw / 2, y), 'b': (x + bw / 2, y + bh)}[side]

    labels = []
    for src, dst, label, s1, s2 in edges:
        (x1, y1), (x2, y2) = anchor(src, s1), anchor(dst, s2)
        if s1 in 'lr' and s2 in 'lr':
            mx = (x1 + x2) / 2
            path = f'M{x1},{y1} L{mx},{y1} L{mx},{y2} L{x2},{y2}'
            lx, ly = mx + 4, (y1 + y2) / 2 - 4
        elif s1 in 'tb' and s2 in 'tb':
            my = (y1 + y2) / 2
            path = f'M{x1},{y1} L{x1},{my} L{x2},{my} L{x2},{y2}'
            lx, ly = (x1 + x2) / 2 + 4, my - 5
        else:
            if s1 in 'lr':        # horizontal first, then vertical
                path = f'M{x1},{y1} L{x2},{y1} L{x2},{y2}'
            else:                 # vertical first, then horizontal
                path = f'M{x1},{y1} L{x1},{y2} L{x2},{y2}'
            lx, ly = (x1 + x2) / 2 + 4, (y1 + y2) / 2 - 4
        p.append(f'<path d="{path}" fill="none" stroke="#333" stroke-width="1.6" marker-end="url(#arr)"/>')
        if label:
            anchor_mid = s1 in 'lr' and s2 in 'lr' and abs(y1 - y2) < 1
            if anchor_mid:
                lx, ly = (x1 + x2) / 2, y1 - 7
            labels.append(f'<text x="{lx:.0f}" y="{ly:.0f}" font-size="11.5" fill="#333" '
                          f'text-anchor="{"middle" if anchor_mid else "start"}" '
                          f'paint-order="stroke" stroke="white" stroke-width="4">{escape(label)}</text>')
    for n, (x, y, bw, bh, label, kind) in nodes.items():
        p.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="9" fill="{COL[kind]}" '
                 f'stroke="{EDGE[kind]}" stroke-width="1.8"/>')
        lines = label.split('\n')
        for i, line in enumerate(lines):
            bold = ' font-weight="bold"' if i == 0 else ' fill="#444" font-size="12"'
            yy = y + bh / 2 + (i - (len(lines) - 1) / 2) * 16 + 4
            p.append(f'<text x="{x + bw/2}" y="{yy:.0f}" text-anchor="middle"{bold}>{escape(line)}</text>')
    p.extend(labels)
    for i, (kind, text) in enumerate(notes):
        p.append(f'<rect x="30" y="{h-30-22*(len(notes)-1-i)}" width="14" height="12" fill="{COL[kind]}" '
                 f'stroke="{EDGE[kind]}"/><text x="50" y="{h-20-22*(len(notes)-1-i)}" font-size="12">{escape(text)}</text>')
    with open(os.path.join(OUT, name), 'w') as fh:
        fh.write('\n'.join(p + ['</svg>']))


LEGEND = [('sim', 'simulation'), ('ros', 'ROS 2 nodes (this repo)'), ('nav', 'Nav2 / SLAM (external)'),
          ('rl', 'reinforcement learning')]

# 1. System overview ─────────────────────────────────────────────────────
diagram('01_system_overview.svg', 'System architecture: data and control flow', 1100, 560, {
    'gz': (30, 80, 210, 90, 'Gazebo Classic\narena + 2 TurtleBot3\n360° lidar, diff drive', 'sim'),
    'slam': (350, 80, 190, 90, 'SLAM (×2)\nslam_toolbox per robot\n/robotN/map', 'nav'),
    'merge': (650, 80, 200, 90, 'Map merge\nmap_merge_node\n→ /map', 'ros'),
    'brain': (650, 260, 200, 110, 'Exploration brain\nheuristic: frontier_coordinator\nRL: ros_policy_node', 'ros'),
    'nav2': (350, 260, 190, 110, 'Nav2 (×2)\nplanner + controller\nrecoveries', 'nav'),
    'policy': (930, 260, 150, 110, 'PPO policy\nmodels/ppo_frontier\n_policy.zip', 'rl'),
    'sim2d': (900, 440, 180, 80, 'rl_sim (training)\n2D sim + MaskablePPO', 'rl'),
}, [
    ('gz', 'slam', 'scan + odom', 'r', 'l'),
    ('slam', 'merge', '/robotN/map', 'r', 'l'),
    ('merge', 'brain', '/map', 'b', 't'),
    ('brain', 'nav2', 'NavigateToPose', 'l', 'r'),
    ('nav2', 'gz', '/robotN/cmd_vel', 'l', 'b'),
    ('policy', 'brain', 'action', 'l', 'r'),
    ('sim2d', 'policy', 'trains', 't', 'b'),
], LEGEND)

# 2. ROS 2 computation graph ─────────────────────────────────────────────
diagram('02_ros_graph.svg', 'ROS 2 graph: nodes, topics and actions (per robot N = 1, 2)', 1100, 600, {
    'gz': (40, 250, 180, 100, 'gzserver\ndiff_drive + lidar plugins\nrobot_state_publisher', 'sim'),
    'scan': (290, 120, 170, 50, '/robotN/scan', 'io'),
    'slam': (530, 100, 180, 90, 'slam_toolbox\n(robotN namespace)', 'nav'),
    'rmap': (530, 230, 180, 50, '/robotN/map', 'io'),
    'merge': (780, 205, 180, 100, 'map_merge_node\nfuses maps\nstatic TF map→robotN/map', 'ros'),
    'map': (800, 360, 140, 50, '/map', 'io'),
    'brain': (760, 450, 220, 90, 'frontier_coordinator\nor rl_policy_coordinator', 'ros'),
    'act': (470, 470, 210, 50, '/robotN/navigate_to_pose', 'io'),
    'nav2': (250, 420, 170, 110, 'Nav2 stack\nbt_navigator, planner,\ncontroller, smoother', 'nav'),
    'cmd': (290, 300, 170, 50, '/robotN/cmd_vel', 'io'),
}, [
    ('gz', 'scan', '', 'r', 'l'), ('scan', 'slam', '', 'r', 'l'), ('slam', 'rmap', '', 'b', 't'),
    ('rmap', 'merge', '', 'r', 'l'), ('merge', 'map', '', 'b', 't'), ('map', 'brain', '', 'b', 't'),
    ('brain', 'act', 'goal', 'l', 'r'), ('act', 'nav2', '', 'l', 'r'),
    ('nav2', 'cmd', '', 't', 'b'), ('cmd', 'gz', '', 'l', 'r'),
], [('io', 'topic / action'), ('ros', 'node in this repo'), ('nav', 'external ROS package'),
    ('sim', 'Gazebo')])

# 3. TF tree ─────────────────────────────────────────────────────────────
diagram('03_tf_tree.svg', 'TF tree', 820, 560, {
    'map': (330, 70, 160, 50, 'map\n(merged frame)', 'ros'),
    'm1': (140, 170, 170, 50, 'robot1/map', 'ros'), 'm2': (510, 170, 170, 50, 'robot2/map', 'ros'),
    'o1': (140, 260, 170, 50, 'robot1/odom', 'nav'), 'o2': (510, 260, 170, 50, 'robot2/odom', 'nav'),
    'b1': (140, 350, 170, 50, 'robot1/base_footprint', 'sim'), 'b2': (510, 350, 170, 50, 'robot2/base_footprint', 'sim'),
    's1': (140, 440, 170, 50, 'robot1/base_scan', 'sim'), 's2': (510, 440, 170, 50, 'robot2/base_scan', 'sim'),
}, [('map', 'm1', 'static (map_merge_node)', 'b', 't'), ('map', 'm2', '', 'b', 't'),
    ('m1', 'o1', 'slam_toolbox', 'b', 't'), ('m2', 'o2', 'slam_toolbox', 'b', 't'),
    ('o1', 'b1', 'Gazebo diff_drive', 'b', 't'), ('o2', 'b2', 'Gazebo diff_drive', 'b', 't'),
    ('b1', 's1', 'robot_state_publisher', 'b', 't'), ('b2', 's2', 'robot_state_publisher', 'b', 't')])

# 4. RL pipeline: train in 2D, deploy in Gazebo ─────────────────────────
diagram('04_rl_pipeline.svg', 'RL pipeline: train in the 2D simulator, deploy zero-shot to Gazebo', 1100, 520, {
    'gen': (40, 90, 200, 80, 'generate_random_world\n(same arena file\nas Gazebo)', 'sim'),
    'world': (300, 90, 200, 80, 'core/world.py\nrasterise 0.05 m grid\n+ inflated nav map', 'rl'),
    'env': (560, 90, 220, 80, 'FrontierExplorationEnv\nlidar · frontiers · A* motion', 'rl'),
    'ppo': (840, 90, 200, 80, 'MaskablePPO\n8 parallel envs\n300k decisions', 'rl'),
    'model': (840, 260, 200, 70, 'ppo_frontier_policy.zip', 'rl'),
    'node': (560, 260, 220, 80, 'ros_policy_node.py\nreal state → same env code', 'ros'),
    'stack': (300, 260, 200, 80, 'Gazebo + SLAM\n+ map merge + Nav2', 'nav'),
    'eval': (560, 410, 220, 70, 'gazebo_test.sh → results/\nanalyze · make_graphs', 'io'),
}, [('gen', 'world', 'SDF', 'r', 'l'), ('world', 'env', '', 'r', 'l'), ('env', 'ppo', 'obs / reward', 'r', 'l'),
    ('ppo', 'model', 'best model', 'b', 't'), ('model', 'node', 'load', 'l', 'r'),
    ('stack', 'node', '/map, TF, results', 'r', 'l'), ('node', 'eval', 'coverage log', 'b', 't')], LEGEND)

# 5. Function-level: one decision of the RL node ─────────────────────────
diagram('05_policy_node_tick.svg', 'Function level: ros_policy_node._tick() (runs every 1 s)', 1100, 560, {
    'ready': (40, 80, 230, 70, 'Nav2 servers ready,\n/map and TF available?', 'dec'),
    'sync': (330, 80, 230, 70, '_resample_map(/map) → belief\nTF → robot cell, heading', 'ros'),
    'life': (620, 80, 230, 70, 'goal lifecycle per robot\nsucceeded / failed / explored / 90 s', 'ros'),
    'front': (880, 200, 190, 70, 'detect_frontiers()\n(coordinator algorithm)', 'rl'),
    'cand': (620, 200, 230, 70, 'env._candidates()\n12 nearest reachable, masked', 'rl'),
    'feat': (330, 200, 230, 70, 'env._decision()\n153 features + action mask', 'rl'),
    'act': (40, 200, 230, 70, 'policy.predict(obs, mask)\nor heuristic cost', 'rl'),
    'send': (40, 330, 230, 70, '_send(): NavigateToPose\nfrontier → world (x, y)', 'ros'),
    'nav': (330, 330, 230, 70, 'Nav2 plans + drives\nresult callback → done', 'nav'),
    'log': (620, 330, 230, 70, 'CSV: sim time, known m²,\ndecisions, failures', 'io'),
}, [('ready', 'sync', 'yes', 'r', 'l'), ('sync', 'life', '', 'r', 'l'), ('life', 'front', 'idle robots', 'r', 't'),
    ('front', 'cand', '', 'l', 'r'), ('cand', 'feat', '', 'l', 'r'), ('feat', 'act', '', 'l', 'r'),
    ('act', 'send', 'slot k', 'b', 't'), ('send', 'nav', '', 'r', 'l'), ('nav', 'log', '', 'r', 'l')],
    [('dec', 'check'), ('ros', 'ROS I/O'), ('rl', 'shared with training (rl_sim)'), ('nav', 'Nav2')])

# 5b. Slide version of 5: input → policy → action ─────────────────────────
diagram('05b_policy_node_simple.svg', 'ros_policy_node: one decision every 1 s', 1100, 560, {
    'in': (40, 170, 300, 150, 'Live ROS 2 state\nmerged /map from SLAM\nrobot poses from TF\nNav2 goal status', 'ros'),
    'pol': (400, 150, 300, 190, 'Same env code as training\n/map resampled to training grid\n12 frontier candidates, masked\n153-feature observation\npolicy.predict() picks one', 'rl'),
    'out': (760, 170, 300, 150, 'Navigation action\nfrontier → world (x, y)\nNavigateToPose goal\nNav2 plans and drives', 'nav'),
}, [('in', 'pol', '', 'r', 'l'), ('pol', 'out', 'slot k', 'r', 'l')],
    [('ros', 'ROS I/O'), ('rl', 'shared with training (rl_sim): no reimplementation drift'), ('nav', 'Nav2')])

# 6. Function-level: training environment step ──────────────────────────
diagram('06_env_step.svg', 'Function level: FrontierExplorationEnv.step(action)', 1100, 560, {
    'act': (40, 90, 220, 70, 'action k (masked slot)\n→ candidate goal cell', 'rl'),
    'astar': (310, 90, 220, 70, 'astar(nav_free, cell, goal)\npath on inflated map', 'rl'),
    'adv': (580, 90, 220, 70, '_advance(): tick 0.5 s\nmove 0.15 m/s · lidar scan', 'sim'),
    'idle': (850, 90, 210, 70, 'robot idle?\n(path done / area explored)', 'dec'),
    'dec': (850, 240, 210, 70, '_decision() for idle robot\nfrontiers → features', 'rl'),
    'rew': (580, 240, 220, 70, 'reward = 0.1·Δknown m²\n− 0.01·Δt', 'rl'),
    'end': (310, 240, 220, 70, 'done? no frontiers left\nor 600 s', 'dec'),
    'obs': (40, 240, 220, 70, 'return obs, reward,\nmask for next decision', 'io'),
}, [('act', 'astar', '', 'r', 'l'), ('astar', 'adv', '', 'r', 'l'), ('adv', 'idle', '', 'r', 'l'),
    ('idle', 'dec', 'yes', 'b', 't'), ('dec', 'rew', '', 'l', 'r'), ('rew', 'end', '', 'l', 'r'),
    ('end', 'obs', '', 'l', 'r')], [('rl', 'decision / learning'), ('sim', 'physics stand-in'), ('dec', 'check')])

# 7. Function-level: heuristic coordinator replan loop ───────────────────
diagram('07_coordinator_replan.svg', 'Function level: frontier_coordinator._replan() (every 3 s)', 1100, 470, {
    'map': (40, 90, 220, 70, '_detect_frontiers(/map)\nmorphology + components', 'ros'),
    'dedup': (310, 90, 220, 70, '_deduplicate()\nmerge within 1.5 m', 'ros'),
    'bl': (580, 90, 220, 70, 'drop blacklisted\n(failed < 30 s ago)', 'ros'),
    'need': (850, 90, 210, 70, 'robot needs goal?\nstuck / reached / vanished', 'dec'),
    'cost': (850, 240, 210, 80, 'cost = distance\n+ 50/d(other goal)\n+ 50·other half', 'rl'),
    'pick': (580, 240, 220, 70, 'lowest cost frontier', 'ros'),
    'send': (310, 240, 220, 70, '_send_nav_goal()\nNavigateToPose', 'nav'),
    'mark': (40, 240, 220, 70, 'publish RViz markers', 'io'),
}, [('map', 'dedup', '', 'r', 'l'), ('dedup', 'bl', '', 'r', 'l'), ('bl', 'need', '', 'r', 'l'),
    ('need', 'cost', 'yes', 'b', 't'), ('cost', 'pick', '', 'l', 'r'), ('pick', 'send', '', 'l', 'r'),
    ('send', 'mark', '', 'l', 'r')], [('ros', 'coordinator logic'), ('rl', 'heuristic cost'), ('nav', 'Nav2'),
                                     ('dec', 'check')])

print('Wrote diagrams to', OUT)
