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


def diagram(name, title, w, h, nodes, edges, notes=(), raw=()):
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
    p.extend(raw)
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

# 1b. System overview, top-down (slide version) ───────────────────────────
STAGE = '<text x="40" y="{}" font-size="13" font-weight="bold" fill="#888">{}</text>'
R = [24, 134, 244, 354, 464]          # row tops, evenly spaced; boxes are 52 high
BH, MID = 52, 26
diagram('../../presentation/system_architecture_topdown.svg', '',
        1100, 560, {
    'gz': (250, R[0], 440, BH, 'Gazebo Classic\narena + robot1, robot2 (lidar, diff drive)', 'sim'),
    's1': (250, R[1], 200, BH, 'slam_toolbox (robot1)\n/robot1/map', 'nav'),
    's2': (490, R[1], 200, BH, 'slam_toolbox (robot2)\n/robot2/map', 'nav'),
    'merge': (330, R[2], 280, BH, 'map_merge_node\nboth maps → one /map', 'ros'),
    'brain': (330, R[3], 280, BH, 'Exploration brain: picks frontiers\nheuristic coordinator  or  RL policy node', 'ros'),
    'pol': (800, R[3], 220, BH, 'PPO policy\nppo_frontier_policy.zip', 'rl'),
    'sim2d': (800, R[4], 220, BH, 'rl_sim\n2D sim + MaskablePPO', 'rl'),
    'n1': (250, R[4], 200, BH, 'Nav2 (robot1)\nplan + drive + recover', 'nav'),
    'n2': (490, R[4], 200, BH, 'Nav2 (robot2)\nplan + drive + recover', 'nav'),
}, [
    ('gz', 's1', '', 'b', 't'), ('gz', 's2', '', 'b', 't'),
    ('s1', 'merge', '', 'b', 't'), ('s2', 'merge', '', 'b', 't'),
    ('merge', 'brain', '/map (2 Hz)', 'b', 't'),
    ('brain', 'n1', 'NavigateToPose goals', 'b', 't'), ('brain', 'n2', '', 'b', 't'),
    ('pol', 'brain', 'chosen frontier', 'l', 'r'), ('sim2d', 'pol', '', 't', 'b'),
], raw=[
    # "scan + odom", centred on the fork under Gazebo
    f'<rect x="432" y="{(R[0]+BH+R[1])//2-17}" width="76" height="16" fill="white"/>',
    f'<text x="470" y="{(R[0]+BH+R[1])//2-5}" font-size="11.5" fill="#333" text-anchor="middle" '
    'paint-order="stroke" stroke="white" stroke-width="4">scan + odom</text>',
    # control loop: both robots' Nav2 velocity commands back up into Gazebo
    f'<path d="M590,{R[4]+BH} L590,{R[4]+BH+23} L160,{R[4]+BH+23} L160,{R[4]+MID}" fill="none" '
    'stroke="#B26A00" stroke-width="2" stroke-dasharray="7 4"/>',
    f'<path d="M250,{R[4]+MID} L160,{R[4]+MID} L160,{R[0]+MID} L250,{R[0]+MID}" fill="none" stroke="#B26A00" '
    'stroke-width="2" stroke-dasharray="7 4" marker-end="url(#arr)"/>',
    f'<text x="150" y="{(R[0]+R[4])//2+MID}" font-size="12" fill="#B26A00" text-anchor="middle" '
    f'transform="rotate(-90 150 {(R[0]+R[4])//2+MID})">/robotN/cmd_vel</text>',
    # legend, top right
    '<rect x="850" y="24" width="220" height="140" rx="8" fill="white" stroke="#ccc"/>',
] + [STAGE.format(r + 31, t) for r, t in
     zip(R, ['1 · SIMULATE', '2 · MAP', '3 · MERGE', '4 · DECIDE', '5 · ACT'])]
  + [f'<rect x="866" y="{y}" width="14" height="12" rx="2" fill="{COL[k]}" stroke="{EDGE[k]}"/>'
     f'<text x="890" y="{y + 10}" font-size="12">{t}</text>' for y, k, t in
     [(38, 'sim', 'simulation'), (62, 'nav', 'SLAM / Nav2 (external)'), (86, 'ros', 'our ROS 2 nodes'),
      (110, 'rl', 'reinforcement learning')]]
  + ['<path d="M866,140 L880,140" stroke="#B26A00" stroke-width="2" stroke-dasharray="4 2"/>'
     '<text x="890" y="144" font-size="12">velocity commands (loop)</text>'])

# 1c. Nav2 inside one robot (slide version) ─────────────────────────────
A = 'fill="none" stroke="#333" stroke-width="1.6" marker-end="url(#arr)"'
T = 'font-size="11.5" fill="#333" paint-order="stroke" stroke="white" stroke-width="4"'
diagram('../../presentation/nav2_architecture.svg', '',
        1100, 560, {
    # three aligned columns: x-centres 270 (planning side), 550 (main line), 845 (robot side)
    'merge': (170, 75, 200, 52, 'SLAM + map merge\nmerged occupancy grid', 'ros'),
    'brain': (450, 75, 200, 52, 'Exploration brain\npicks the next frontier', 'ros'),
    'bt': (420, 180, 260, 52, 'BT Navigator\nplan → drive → recover if stuck', 'nav'),
    'planner': (155, 300, 230, 52, 'Planner\nglobal path to the frontier', 'nav'),
    'ctrl': (425, 300, 250, 52, 'Controller\nfollows the path, avoids obstacles', 'nav'),
    'beh': (745, 300, 200, 52, 'Recovery\nspin · back up · wait', 'nav'),
    'gcm': (155, 410, 230, 52, 'Global costmap\nwhole arena · 1 Hz', 'nav'),
    'lcm': (425, 410, 250, 52, 'Local costmap\n3 × 3 m around the robot · 5 Hz', 'nav'),
    'gz': (745, 410, 200, 52, 'Robot in Gazebo\nwheels move, lidar scans', 'sim'),
}, [
    ('merge', 'brain', '/map', 'r', 'l'),
    ('brain', 'bt', 'NavigateToPose (x, y)', 'b', 't'),
    ('bt', 'planner', '', 'b', 't'), ('bt', 'ctrl', '', 'b', 't'), ('bt', 'beh', '', 'b', 't'),
    ('planner', 'ctrl', 'path', 'r', 'l'),
    ('gcm', 'planner', '', 't', 'b'), ('lcm', 'ctrl', '', 't', 'b'),
], raw=[
    f'<path d="M490,180 L490,127" {A}/>', f'<text x="482" y="158" {T} text-anchor="end">succeeded / failed</text>',
    f'<path d="M675,326 L710,326 L710,436 L745,436" {A}/>', f'<text x="715" y="390" {T}>cmd_vel</text>',
    f'<path d="M845,462 L845,500 L270,500 L270,462" {A}/>', f'<path d="M550,500 L550,462" {A}/>',
    f'<text x="700" y="504" {T} text-anchor="middle">lidar scan</text>',
])

# 3b. Frames and map merging as data flow (slide version) ────────────────
diagram('../../presentation/map_merging_flow.svg', 'Frames and map merging: how /map is built', 820, 560, {
    'g1': (95, 60, 260, 52, 'robot1 in Gazebo\nlidar + wheel odometry', 'sim'),
    'g2': (465, 60, 260, 52, 'robot2 in Gazebo\nlidar + wheel odometry', 'sim'),
    's1': (75, 165, 300, 68, 'slam_toolbox (robot1)\nbuilds /robot1/map\nTF robot1/map → robot1/odom', 'nav'),
    's2': (445, 165, 300, 68, 'slam_toolbox (robot2)\nbuilds /robot2/map\nTF robot2/map → robot2/odom', 'nav'),
    'merge': (240, 290, 340, 68, 'map_merge_node (2 Hz)\nreprojects both maps into one grid\nstatic TF map → robotN/map (offset 0)', 'ros'),
    'map': (290, 410, 240, 52, '/map (merged)\nused by the exploration brain', 'ros'),
}, [
    ('g1', 's1', '/robot1/scan, odom', 'b', 't'), ('g2', 's2', '/robot2/scan, odom', 'b', 't'),
    ('s1', 'merge', '/robot1/map', 'b', 't'), ('s2', 'merge', '/robot2/map', 'b', 't'),
    ('merge', 'map', '', 'b', 't'),
], raw=[
    '<text x="410" y="500" text-anchor="middle" font-size="12" fill="#444">TF chain per robot: '
    'map → robotN/map → robotN/odom → robotN/base_footprint → robotN/base_scan</text>',
    '<text x="410" y="522" text-anchor="middle" font-size="12" fill="#444">Offsets are 0: Gazebo odometry '
    'starts at each robot\'s spawn pose, so each SLAM map is already in world coordinates</text>',
])

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
diagram('05b_policy_node_simple.svg', 'ros_policy_node: from the Gazebo world to a frontier choice (every 1 s)', 1100, 560, {
    'gz': (40, 60, 230, 70, 'Gazebo 3D world\n2 TurtleBot3s: lidar + odometry', 'sim'),
    'map': (310, 60, 230, 70, 'SLAM + map merge\n2D /map, robot poses from TF', 'nav'),
    'grid': (580, 60, 230, 70, 'Resample to training grid\n0.05 m cells, as in rl_sim', 'ros'),
    'cand': (850, 60, 210, 70, 'Frontier candidates\n12 nearest reachable', 'rl'),
    'per': (40, 175, 270, 135, 'Per candidate: 12 numbers\nposition, distance, size\nunknown area around it\ndistance to other robot + its goal\nheading, heuristic\'s pick', 'rl'),
    'glob': (40, 335, 270, 105, 'Global: 9 numbers\nboth robot positions\nother robot\'s goal\n% mapped, elapsed time', 'rl'),
    'obs': (380, 265, 200, 80, '153-D observation\n12 × 12 + 9', 'rl'),
    'ppo': (630, 265, 200, 80, 'PPO policy\nscores the 12 slots\n(empty slots masked)', 'rl'),
    'goal': (880, 265, 190, 80, 'Best frontier\nslot k → world (x, y)\nNavigateToPose → Nav2', 'nav'),
}, [('gz', 'map', '', 'r', 'l'), ('map', 'grid', '', 'r', 'l'), ('grid', 'cand', '', 'r', 'l'),
    ('cand', 'per', 'for the robot that needs a goal', 'b', 't'),
    ('per', 'obs', '', 'r', 'l'), ('glob', 'obs', '', 'r', 'l'),
    ('obs', 'ppo', '', 'r', 'l'), ('ppo', 'goal', 'slot k', 'r', 'l')],
    [('sim', 'Gazebo'), ('nav', 'SLAM / Nav2'), ('ros', 'ROS glue'),
     ('rl', 'same env code as training (no reimplementation drift)')])

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
