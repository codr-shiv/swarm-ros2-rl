"""
Slide graphic (v2): the RL formulation in four panels.

  python3 docs/architecture/make_rl_design_2.py   # writes presentation/rl_design_choices_2.svg

Throughput numbers (panel 4):
  2D sim: median 193 decisions/s over the training run (time/fps, 8 parallel envs).
  Gazebo: 18-23 decisions per 240 s run in every logged run (results/), so even
          8 headless stacks in parallel at real time give ~8 × 20 / 240 ≈ 0.7 decisions/s.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'presentation', 'rl_design_choices_2.svg')
W, H = 1100, 560
PW, PH = 515, 228
RL, RL_E = '#F3E8FA', '#7B3FA6'
SIM, SIM_E = '#E8F1FB', '#1F5FA8'
ROS, ROS_E = '#EAF6EC', '#2E7D32'
NAV, NAV_E = '#FFF4E0', '#B26A00'
GREY = '#666'

p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="13">',
     '<defs><marker id="arr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
     '<path d="M0,0 L10,4 L0,8 z" fill="#333"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="32" text-anchor="middle" font-size="20" font-weight="bold">'
     'RL brain: how the decision is learned</text>']


def panel(x, y, title, sub=None):
    p.append(f'<rect x="{x}" y="{y}" width="{PW}" height="{PH}" rx="12" fill="#FAFAFA" stroke="#DDD" stroke-width="1.5"/>')
    p.append(f'<text x="{x+18}" y="{y+30}" font-size="16" font-weight="bold">{title}</text>')
    if sub:
        p.append(f'<text x="{x+18}" y="{y+50}" font-size="12.5" fill="{GREY}">{sub}</text>')


def text(x, y, s, size=12.5, anchor='start', bold=False, fill='#333'):
    b = ' font-weight="bold"' if bold else ''
    p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"{b}>{s}</text>')


def box(bx, by, bw, bh, t1, t2=None, fill=RL, edge=RL_E, size2=11):
    p.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" fill="{fill}" stroke="{edge}" stroke-width="1.6"/>')
    if t2 is None:
        text(bx + bw / 2, by + bh / 2 + 5, t1, anchor='middle', bold=True)
    else:
        text(bx + bw / 2, by + bh / 2 - 3, t1, anchor='middle', bold=True)
        text(bx + bw / 2, by + bh / 2 + 13, t2, anchor='middle', size=size2, fill=GREY)


def arrow(x1, y1, x2, y2):
    p.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#333" stroke-width="1.6" marker-end="url(#arr)"/>')


def tag(x, y, label, fill, edge):
    p.append(f'<rect x="{x}" y="{y}" width="82" height="22" rx="11" fill="{fill}" stroke="{edge}" stroke-width="1.4"/>')
    text(x + 41, y + 15, label, size=11, anchor='middle', bold=True, fill=edge)


# 1. Waypoints, not velocities ──────────────────────────────────────────
x, y = 30, 60
panel(x, y, '1 · Pick waypoints, not velocities', 'one action = one whole trip to a frontier')
for i in range(70):                                   # dense motor commands
    xx = x + 24 + i * 4.6
    p.append(f'<line x1="{xx:.1f}" y1="{y+88}" x2="{xx:.1f}" y2="{y+108}" stroke="#BBB" stroke-width="1.6"/>')
text(x + 360, y + 96, 'velocity commands', bold=True, fill=GREY)
text(x + 360, y + 113, 'hundreds per episode', fill=GREY)
pts = [(x + 30 + i * 34, y + 172 + (10 if i % 2 else -10)) for i in range(10)]
p.append('<polyline points="' + ' '.join(f'{a},{b}' for a, b in pts) +
         f'" fill="none" stroke="{RL_E}" stroke-width="2" stroke-dasharray="5 4"/>')
for i, (a, b) in enumerate(pts):
    p.append(f'<circle cx="{a}" cy="{b}" r="9" fill="{RL}" stroke="{RL_E}" stroke-width="2"/>')
    text(a, b + 4, str(i + 1), size=10, anchor='middle', bold=True, fill=RL_E)
text(x + 360, y + 168, 'frontier choices', bold=True, fill=RL_E)
text(x + 360, y + 185, '≈ 10 per episode', fill=RL_E)

# 2. Network + PPO ──────────────────────────────────────────────────────
x, y = 555, 60
panel(x, y, '2 · Actor-critic network, trained with PPO', 'MaskablePPO · separate 2 × 128 tanh MLPs · 74k parameters')
cy = y + 102
box(x + 18, cy - 22, 62, 44, '153', 'obs', SIM, SIM_E)
box(x + 98, cy - 32, 48, 64, '128')
box(x + 160, cy - 32, 48, 64, '128')
box(x + 250, cy - 44, 247, 38, 'Actor: 12 slot scores', 'mask invalid → softmax → pick slot')
box(x + 250, cy + 6, 247, 38, 'Critic: value of the state', 'expected return, used for advantages')
arrow(x + 80, cy, x + 98, cy); arrow(x + 146, cy, x + 160, cy)
arrow(x + 208, cy - 8, x + 250, cy - 25); arrow(x + 208, cy + 8, x + 250, cy + 25)
# training loop
ly = y + 160
box(x + 18, ly, 150, 42, 'Collect', '2,048 decisions / update', ROS, ROS_E, 10.5)
box(x + 183, ly, 150, 42, 'Advantages (GAE)', 'γ = 0.99 · λ = 0.95', ROS, ROS_E, 10.5)
box(x + 348, ly, 150, 42, 'Clipped update', '10 epochs · ε = 0.2', ROS, ROS_E, 10.5)
arrow(x + 168, ly + 21, x + 183, ly + 21); arrow(x + 333, ly + 21, x + 348, ly + 21)
p.append(f'<path d="M{x+423},{ly+42} L{x+423},{ly+54} L{x+93},{ly+54} L{x+93},{ly+42}" fill="none" '
         f'stroke="#333" stroke-width="1.4" marker-end="url(#arr)"/>')
text(x + 258, ly + 58, 'repeat ≈ 146 times', size=10.5, anchor='middle', fill=GREY)
p[-1] = p[-1].replace('<text ', '<text paint-order="stroke" stroke="#FAFAFA" stroke-width="5" ')

# 3. State, action, reward ──────────────────────────────────────────────
x, y = 30, 315
panel(x, y, '3 · What the policy sees, does and earns')
tag(x + 18, y + 48, 'OBSERVE', SIM, SIM_E)
text(x + 112, y + 64, '153 numbers: 12 nearest frontiers × 12 features + 9 global', bold=True, size=12)
text(x + 112, y + 82, 'per frontier: position, distance, size, unknown area nearby,', size=11.5, fill=GREY)
text(x + 112, y + 98, 'distance to the other robot and its goal, heading, heuristic’s pick', size=11.5, fill=GREY)
text(x + 112, y + 114, 'global: both robot positions, other robot’s goal, % explored, time', size=11.5, fill=GREY)
tag(x + 18, y + 132, 'ACT', RL, RL_E)
text(x + 112, y + 148, 'pick 1 of the 12 frontiers (slot k)', bold=True, size=12)
text(x + 112, y + 164, 'empty or invalid slots are masked; Nav2 drives the robot there', size=11.5, fill=GREY)
tag(x + 18, y + 182, 'REWARD', NAV, NAV_E)
text(x + 112, y + 198, '+0.1 per newly mapped m²  ·  −0.01 per second', bold=True, size=12)
text(x + 112, y + 214, 'every run maps the same area, so higher return = faster', size=11.5, fill=GREY)

# 4. Fast 2D simulator ──────────────────────────────────────────────────
x, y = 555, 315
panel(x, y, '4 · Train in a fast 2D simulator', 'same arena, frontier detector, speed and clearance as Gazebo')
for cx, name, val, note, fill, edge in [
        (x + 130, '2D simulator', '≈ 190', '8 parallel envs (measured)', RL, RL_E),
        (x + 385, 'Gazebo, headless', '≈ 0.7', '8 parallel stacks, best case', SIM, SIM_E)]:
    p.append(f'<rect x="{cx-110}" y="{y+66}" width="220" height="96" rx="10" fill="{fill}" stroke="{edge}" stroke-width="1.6"/>')
    text(cx, y + 88, name, anchor='middle', bold=True, fill=edge)
    text(cx, y + 122, val, anchor='middle', bold=True, size=30, fill=edge)
    text(cx, y + 140, 'decisions / s', anchor='middle', size=11.5, fill=GREY)
    text(cx, y + 155, note, anchor='middle', size=10.5, fill=GREY)
text(x + PW / 2, y + 190, '≈ 300× more decisions per second', anchor='middle', bold=True, size=15, fill=RL_E)
text(x + PW / 2, y + 210, 'policy converged after ≈ 30k decisions · Gazebo: ≈ 20 decisions per 240 s run',
     anchor='middle', size=11, fill=GREY)

with open(OUT, 'w') as fh:
    fh.write('\n'.join(p + ['</svg>']))
print('Wrote', os.path.normpath(OUT))
