"""
Slide graphic for "The 2D training simulator": waypoints vs velocities, simulator speed, tech stack.

  python3 docs/architecture/make_sim_slide.py   # writes presentation/rl_simulator_slide.svg

Versions are those installed in the ros-humble distrobox (2026-10). Throughput: see make_rl_design_2.py.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'presentation', 'rl_simulator_slide.svg')
W, H = 1100, 560
RL, RL_E = '#F3E8FA', '#7B3FA6'
SIM, SIM_E = '#E8F1FB', '#1F5FA8'
ROS, ROS_E = '#EAF6EC', '#2E7D32'
NAV, NAV_E = '#FFF4E0', '#B26A00'
GREY = '#666'

p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="13">',
     '<defs><marker id="arr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
     '<path d="M0,0 L10,4 L0,8 z" fill="#333"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']


def panel(x, y, w, h, title, sub=None):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#FAFAFA" stroke="#DDD" stroke-width="1.5"/>')
    p.append(f'<text x="{x+18}" y="{y+30}" font-size="16" font-weight="bold">{title}</text>')
    if sub:
        p.append(f'<text x="{x+18}" y="{y+50}" font-size="12.5" fill="{GREY}">{sub}</text>')


def text(x, y, s, size=12.5, anchor='start', bold=False, fill='#333', extra=''):
    b = ' font-weight="bold"' if bold else ''
    p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"{b}{extra}>{s}</text>')


def box(bx, by, bw, bh, t1, t2=None, fill=RL, edge=RL_E, size1=12.5, size2=10.5):
    p.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" fill="{fill}" stroke="{edge}" stroke-width="1.6"/>')
    if t2 is None:
        text(bx + bw / 2, by + bh / 2 + 5, t1, size=size1, anchor='middle', bold=True)
    else:
        text(bx + bw / 2, by + bh / 2 - 2, t1, size=size1, anchor='middle', bold=True)
        text(bx + bw / 2, by + bh / 2 + 12, t2, size=size2, anchor='middle', fill=GREY)


def path(d, head=True, color='#333', width=1.6):
    m = ' marker-end="url(#arr)"' if head else ''
    p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{m}/>')


def tag(x, y, label, fill, edge, w=112):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="22" rx="11" fill="{fill}" stroke="{edge}" stroke-width="1.4"/>')
    text(x + w / 2, y + 15, label, size=11, anchor='middle', bold=True, fill=edge)


def section(x, y, label):
    text(x, y, label, size=11, bold=True, fill='#999', extra=' letter-spacing="1"')



def item(x, y, name, role):
    p.append(f'<text x="{x}" y="{y}" font-size="12.5"><tspan font-weight="bold" fill="#333">{name}</tspan>'
             f'<tspan dx="7" fill="{GREY}" font-size="11.5">·\u00a0\u00a0{role}</tspan></text>')


# Pick waypoints, not velocities ────────────────────────────────────────
x, y, w, h = 30, 16, 515, 280
panel(x, y, w, h, 'Pick waypoints, not velocities', 'one action = one whole trip to a frontier')
for i in range(70):
    xx = x + 24 + i * 4.6
    p.append(f'<line x1="{xx:.1f}" y1="{y+100}" x2="{xx:.1f}" y2="{y+124}" stroke="#BBB" stroke-width="1.6"/>')
text(x + 362, y + 108, 'velocity commands', bold=True, fill=GREY)
text(x + 362, y + 125, 'hundreds per episode', fill=GREY)
pts = [(x + 32 + i * 34, y + 200 + (12 if i % 2 else -12)) for i in range(10)]
p.append('<polyline points="' + ' '.join(f'{a},{b}' for a, b in pts) +
         f'" fill="none" stroke="{RL_E}" stroke-width="2" stroke-dasharray="5 4"/>')
for i, (a, b) in enumerate(pts):
    p.append(f'<circle cx="{a}" cy="{b}" r="10" fill="{RL}" stroke="{RL_E}" stroke-width="2"/>')
    text(a, b + 4, str(i + 1), size=10, anchor='middle', bold=True, fill=RL_E)
text(x + 362, y + 196, 'frontier choices', bold=True, fill=RL_E)
text(x + 362, y + 213, '≈ 10 per episode', fill=RL_E)
text(x + w / 2, y + 260, 'short episodes → each reward is easy to credit to the choice that earned it',
     size=11.5, anchor='middle', fill=GREY)

# Train in a fast 2D simulator ──────────────────────────────────────────
x, y, w, h = 555, 16, 515, 280
panel(x, y, w, h, 'Train in a fast 2D simulator', 'same arena, frontiers and speed as Gazebo')
for cx, name, val, note, fill, edge in [
        (x + 132, '2D simulator', '≈ 190', '8 parallel envs (measured)', RL, RL_E),
        (x + 383, 'Gazebo, headless', '≈ 0.7', '8 parallel stacks, best case', SIM, SIM_E)]:
    p.append(f'<rect x="{cx-112}" y="{y+76}" width="224" height="112" rx="10" fill="{fill}" stroke="{edge}" stroke-width="1.6"/>')
    text(cx, y + 100, name, anchor='middle', bold=True, fill=edge)
    text(cx, y + 138, val, anchor='middle', bold=True, size=32, fill=edge)
    text(cx, y + 158, 'decisions / s', anchor='middle', size=11.5, fill=GREY)
    text(cx, y + 175, note, anchor='middle', size=10.5, fill=GREY)
text(x + w / 2, y + 226, '≈ 300× more decisions per second', anchor='middle', bold=True, size=16, fill=RL_E)
text(x + w / 2, y + 252, 'Gazebo makes ≈ 20 decisions per 240 s run · the policy needed ≈ 30k to converge',
     anchor='middle', size=11, fill=GREY)

# Tech stack ────────────────────────────────────────────────────────────
x, y, w, h = 30, 310, 1040, 234
panel(x, y, w, h, 'Tech stack')
cols = [
    (x + 18, 'RL TRAINING', RL, RL_E, [
        ('Stable-Baselines3 2.9', 'PPO'),
        ('sb3-contrib 2.9', 'MaskablePPO'),
        ('PyTorch 2.14 (CPU)', 'the networks'),
        ('Gymnasium 1.3', 'environment API'),
        ('TensorBoard', 'training curves')]),
    (x + 365, '2D SIMULATOR (rl_sim)', ROS, ROS_E, [
        ('Python 3.10', 'our own environment'),
        ('NumPy 2.2', 'grids, vectorised lidar'),
        ('OpenCV', 'arena raster, frontiers'),
        ('A* planner', 'stands in for Nav2'),
        ('generate_random_world', 'same arena file')]),
    (x + 712, 'FULL SIMULATION', SIM, SIM_E, [
        ('ROS 2 Humble', 'rclpy policy node'),
        ('Gazebo Classic 11', 'physics + sensors'),
        ('TurtleBot3 Burger ×2', 'LDS-01 lidar'),
        ('slam_toolbox 2.6', 'mapping (×2)'),
        ('Nav2 1.1', 'planning + driving')]),
]
for cx0, label, fill, edge, items in cols:
    tag(cx0, y + 44, label, fill, edge, w=190)
    for i, (name, role) in enumerate(items):
        p.append(f'<circle cx="{cx0+6}" cy="{y+90+i*28}" r="3" fill="{edge}"/>')
        item(cx0 + 16, y + 94 + i * 28, name, role)
for dx in (347, 694):
    p.append(f'<line x1="{x+dx}" y1="{y+44}" x2="{x+dx}" y2="{y+h-18}" stroke="#E2E2E2" stroke-width="1.5"/>')

with open(OUT, 'w') as fh:
    fh.write('\n'.join(p + ['</svg>']))
print('Wrote', os.path.normpath(OUT))
