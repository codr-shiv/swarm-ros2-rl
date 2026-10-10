"""
Slide graphic (v3): the RL formulation, with the network and PPO loop as the main panel.

  python3 docs/architecture/make_rl_design_3.py   # writes presentation/rl_design_choices_3.svg

Throughput numbers (panel 4): see make_rl_design_2.py.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'presentation', 'rl_design_choices_3.svg')
W, H = 1100, 560
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


LX, LW = 30, 430          # left column
RX, RW = 475, 595         # right (main) panel

# 1. Waypoints, not velocities ──────────────────────────────────────────
x, y = LX, 60
panel(x, y, LW, 140, 'Pick waypoints, not velocities', 'one action = one whole trip to a frontier')
for i in range(55):
    xx = x + 22 + i * 4.6
    p.append(f'<line x1="{xx:.1f}" y1="{y+64}" x2="{xx:.1f}" y2="{y+80}" stroke="#BBB" stroke-width="1.6"/>')
text(x + 290, y + 70, 'velocity commands', size=12, bold=True, fill=GREY)
text(x + 290, y + 85, 'hundreds per episode', size=11.5, fill=GREY)
pts = [(x + 30 + i * 26, y + 114 + (7 if i % 2 else -7)) for i in range(10)]
path('M' + ' L'.join(f'{a},{b}' for a, b in pts), head=False, color=RL_E, width=2)
p[-1] = p[-1].replace('stroke-width="2"', 'stroke-width="2" stroke-dasharray="5 4"')
for i, (a, b) in enumerate(pts):
    p.append(f'<circle cx="{a}" cy="{b}" r="8" fill="{RL}" stroke="{RL_E}" stroke-width="2"/>')
    text(a, b + 3.5, str(i + 1), size=9, anchor='middle', bold=True, fill=RL_E)
text(x + 290, y + 110, 'frontier choices', size=12, bold=True, fill=RL_E)
text(x + 290, y + 125, '≈ 10 per episode', size=11.5, fill=RL_E)

# 3. Observation, action, reward ────────────────────────────────────────
x, y = LX, 210
panel(x, y, LW, 166, 'Observation, action, reward')
tag(x + 18, y + 50, 'OBSERVATION', SIM, SIM_E)
text(x + 142, y + 60, '153 numbers per decision', size=12.5, bold=True)
text(x + 142, y + 77, '12 frontiers × 12 features + 9 global', size=12.5, bold=True)
tag(x + 18, y + 96, 'ACTION', RL, RL_E)
text(x + 142, y + 112, 'pick 1 of the 12 nearest frontiers', size=12.5, bold=True)
tag(x + 18, y + 134, 'REWARD', NAV, NAV_E)
text(x + 142, y + 152, 'r = 0.1 · ΔA − 0.01 · Δt', size=16, bold=True, fill='#222',
     extra=' font-family="serif" font-style="italic"')

# 4. Fast 2D simulator ──────────────────────────────────────────────────
x, y = LX, 386
panel(x, y, LW, 144, 'Train in a fast 2D simulator', 'same arena, frontiers and speed as Gazebo')
for bx, name, val, fill, edge in [(x + 18, '2D simulator', '≈ 190', RL, RL_E),
                                  (x + 222, 'Gazebo (8 headless)', '≈ 0.7', SIM, SIM_E)]:
    p.append(f'<rect x="{bx}" y="{y+62}" width="190" height="50" rx="9" fill="{fill}" stroke="{edge}" stroke-width="1.5"/>')
    text(bx + 95, y + 80, name, size=11.5, anchor='middle', bold=True, fill=edge)
    text(bx + 95, y + 102, f'{val} decisions / s', size=15, anchor='middle', bold=True, fill=edge)
text(x + LW / 2, y + 132, '≈ 300× more decisions per second', size=14, anchor='middle', bold=True, fill=RL_E)

# 2. Network + PPO (main panel) ─────────────────────────────────────────
x, y = RX, 60
panel(x, y, RW, 470, 'Actor-critic network, trained with PPO')

# the network
section(x + 18, y + 54, 'THE NETWORK')
cy = y + 128
box(x + 18, cy - 30, 74, 60, '153', 'observation', SIM, SIM_E)
ay, vy = cy - 58, cy + 18           # actor row / critic row (top of boxes)
for rowy in (ay, vy):
    box(x + 118, rowy, 58, 40, '128', 'tanh')
    box(x + 190, rowy, 58, 40, '128', 'tanh')
    path(f'M{x+176},{rowy+20} L{x+190},{rowy+20}')
path(f'M{x+92},{cy} L{x+104},{cy} L{x+104},{ay+20} L{x+118},{ay+20}')
path(f'M{x+104},{cy} L{x+104},{vy+20} L{x+118},{vy+20}')
text(x + 118, ay - 6, 'ACTOR: which frontier?', size=11, bold=True, fill=RL_E)
text(x + 118, vy + 56, 'CRITIC: how good is this situation?', size=11, bold=True, fill=RL_E)
# actor head
box(x + 262, ay, 82, 40, '12 scores', 'one per slot')
box(x + 358, ay, 116, 40, 'mask + softmax', 'invalid slots → 0')
path(f'M{x+248},{ay+20} L{x+262},{ay+20}')
path(f'M{x+344},{ay+20} L{x+358},{ay+20}')
path(f'M{x+474},{ay+20} L{x+488},{ay+20}')
text(x + 492, ay + 17, 'π(slot | s)', size=14, bold=True, fill='#222', extra=' font-family="serif" font-style="italic"')
text(x + 492, ay + 33, 'probabilities', size=10.5, fill=GREY)
# critic head
box(x + 262, vy, 82, 40, 'V(s)', '1 number')
path(f'M{x+248},{vy+20} L{x+262},{vy+20}')
path(f'M{x+344},{vy+20} L{x+358},{vy+20}')
text(x + 362, vy + 17, 'expected return from here', size=12, bold=True)
text(x + 362, vy + 33, 'the baseline to judge actions against', size=10.5, fill=GREY)

# the training loop
section(x + 18, y + 238, 'THE TRAINING LOOP (PPO)')
steps = [
    ('Collect experience', 'run π in 8 simulators × 256 decisions = 2,048 (state, action, reward)'),
    ('Score every decision: advantage', 'Â = (return actually got) − V(s)  →  better or worse than expected?'),
    ('Update the actor (clipped)', 'raise π of slots with Â &gt; 0, lower Â &lt; 0; each by at most ±20 % per update'),
    ('Update the critic', 'fit V(s) to the returns actually received; small entropy bonus keeps exploring'),
]
sy0, sh, gap = y + 252, 42, 12
for i, (name, detail) in enumerate(steps):
    sy = sy0 + i * (sh + gap)
    p.append(f'<rect x="{x+48}" y="{sy}" width="500" height="{sh}" rx="8" fill="{ROS}" stroke="{ROS_E}" stroke-width="1.5"/>')
    p.append(f'<circle cx="{x+30}" cy="{sy+sh/2}" r="11" fill="{ROS_E}"/>')
    text(x + 30, sy + sh / 2 + 4.5, str(i + 1), size=12, anchor='middle', bold=True, fill='white')
    text(x + 60, sy + 18, name, size=12.5, bold=True)
    text(x + 60, sy + 34, detail, size=11, fill='#444')
    if i < len(steps) - 1:
        path(f'M{x+30},{sy+sh/2+11} L{x+30},{sy+sh+gap+sh/2-11}', width=1.4)
# loop back: step 4 -> step 1
top, bot = sy0 + sh / 2, sy0 + 3 * (sh + gap) + sh / 2
path(f'M{x+548},{bot} L{x+568},{bot} L{x+568},{top} L{x+548},{top}', width=1.6)
text(x + 582, (top + bot) / 2, 'repeat ≈ 146×', size=11, anchor='middle', bold=True, fill=ROS_E,
     extra=f' transform="rotate(-90 {x+582} {(top+bot)/2})"')

with open(OUT, 'w') as fh:
    fh.write('\n'.join(p + ['</svg>']))
print('Wrote', os.path.normpath(OUT))
