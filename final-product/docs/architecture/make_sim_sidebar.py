"""
Portrait sidebar for "The 2D training simulator" slide (sits next to the simulator GIF):
waypoints vs velocities, simulator speed, tech stack.

  python3 docs/architecture/make_sim_sidebar.py   # writes presentation/rl_simulator_sidebar.svg

Throughput numbers: see make_rl_design_2.py.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'presentation', 'rl_simulator_sidebar.svg')
W, H = 380, 454
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



def card(x, y, w, h, title, sub=None):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#FAFAFA" stroke="#DDD" stroke-width="1.5"/>')
    text(x + 14, y + 24, title, size=14.5, bold=True)
    if sub:
        text(x + 14, y + 41, sub, size=11, fill=GREY)


X, CW = 6, 368

# Pick waypoints, not velocities
y = 6
card(X, y, CW, 146, 'Pick waypoints, not velocities', 'one action = one whole trip to a frontier')
for i in range(40):
    xx = X + 16 + i * 4.5
    p.append(f'<line x1="{xx:.1f}" y1="{y+60}" x2="{xx:.1f}" y2="{y+78}" stroke="#BBB" stroke-width="1.5"/>')
text(X + 210, y + 68, 'velocity commands', size=11.5, bold=True, fill=GREY)
text(X + 210, y + 82, 'hundreds per episode', size=11, fill=GREY)
pts = [(X + 22 + i * 19, y + 116 + (7 if i % 2 else -7)) for i in range(10)]
p.append('<polyline points="' + ' '.join(f'{a},{b}' for a, b in pts) +
         f'" fill="none" stroke="{RL_E}" stroke-width="1.8" stroke-dasharray="4 3"/>')
for a, b in pts:
    p.append(f'<circle cx="{a}" cy="{b}" r="6.5" fill="{RL}" stroke="{RL_E}" stroke-width="1.8"/>')
text(X + 210, y + 113, 'frontier choices', size=11.5, bold=True, fill=RL_E)
text(X + 210, y + 127, '≈ 10 per episode', size=11, fill=RL_E)

# Train in a fast 2D simulator
y = 162
card(X, y, CW, 156, 'Train in a fast 2D simulator', 'same arena, frontiers and speed as Gazebo')
for bx, name, val, fill, edge in [(X + 14, '2D simulator', '≈ 190', RL, RL_E),
                                  (X + 190, 'Gazebo (8 headless)', '≈ 0.7', SIM, SIM_E)]:
    p.append(f'<rect x="{bx}" y="{y+54}" width="164" height="70" rx="8" fill="{fill}" stroke="{edge}" stroke-width="1.5"/>')
    text(bx + 82, y + 72, name, size=11, anchor='middle', bold=True, fill=edge)
    text(bx + 82, y + 99, val, size=22, anchor='middle', bold=True, fill=edge)
    text(bx + 82, y + 115, 'decisions / s', size=10.5, anchor='middle', fill=GREY)
text(X + CW / 2, y + 144, '≈ 300× more decisions per second', size=13, anchor='middle', bold=True, fill=RL_E)

# Tech stack
y = 328
card(X, y, CW, 120, 'Tech stack')
groups = [('RL TRAINING', RL, RL_E, ['Stable-Baselines3 · MaskablePPO', 'PyTorch · Gymnasium · TensorBoard']),
          ('2D SIMULATOR', ROS, ROS_E, ['NumPy · OpenCV'])]
gy = y + 40
for label, fill, edge, lines in groups:
    tag(X + 14, gy, label, fill, edge, w=112)
    for i, line in enumerate(lines):
        text(X + 138, gy + 15 + i * 16, line, size=11.5, bold=True)
    gy += max(len(lines) * 16, 22) + 12

with open(OUT, 'w') as fh:
    fh.write('\n'.join(p + ['</svg>']))
print('Wrote', os.path.normpath(OUT))
