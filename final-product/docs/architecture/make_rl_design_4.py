"""
Slide graphic (v4): observation/action/reward strip + the actor-critic network and PPO loop.

  python3 docs/architecture/make_rl_design_4.py   # writes presentation/rl_design_choices_4.svg
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'presentation', 'rl_design_choices_4.svg')
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


# Observation, action, reward (top strip) ───────────────────────────────
x, y, w = 30, 60, 1040
p.append(f'<rect x="{x}" y="{y}" width="{w}" height="96" rx="12" fill="#FAFAFA" stroke="#DDD" stroke-width="1.5"/>')
cols = [x + w / 6, x + w / 2, x + 5 * w / 6]
tag(cols[0] - 66, y + 16, 'OBSERVATION', SIM, SIM_E, w=132)
text(cols[0], y + 62, '153 numbers per decision', size=13, bold=True, anchor='middle')
text(cols[0], y + 79, '12 frontiers × 12 features + 9 global', size=13, bold=True, anchor='middle')
tag(cols[1] - 66, y + 16, 'ACTION', RL, RL_E, w=132)
text(cols[1], y + 62, 'pick 1 of the 12', size=13, bold=True, anchor='middle')
text(cols[1], y + 79, 'nearest frontiers', size=13, bold=True, anchor='middle')
tag(cols[2] - 66, y + 16, 'REWARD', NAV, NAV_E, w=132)
text(cols[2], y + 74, 'r = 0.1 · ΔA − 0.01 · Δt', size=19, bold=True, fill='#222', anchor='middle',
     extra=' font-family="serif" font-style="italic"')
for dx in (w / 3, 2 * w / 3):
    p.append(f'<line x1="{x+dx}" y1="{y+16}" x2="{x+dx}" y2="{y+80}" stroke="#E2E2E2" stroke-width="1.5"/>')

# Actor-critic network, trained with PPO ────────────────────────────────
x, y, w, h = 30, 170, 1040, 378
panel(x, y, w, h, 'Actor-critic network, trained with PPO')
p.append(f'<line x1="{x+590}" y1="{y+48}" x2="{x+590}" y2="{y+h-20}" stroke="#E2E2E2" stroke-width="1.5"/>')

# the network (left), top to bottom
section(x + 18, y + 56, 'THE NETWORK')
ax, cx_ = x + 160, x + 390          # actor / critic column centres
box(x + 215, y + 52, 120, 44, '153', 'observation', SIM, SIM_E)
path(f'M{x+215},{y+74} L{ax},{y+74} L{ax},{y+116}')
path(f'M{x+335},{y+74} L{cx_},{y+74} L{cx_},{y+116}')
text(ax - 10, y + 92, 'ACTOR', size=11.5, bold=True, fill=RL_E, anchor='end')
text(ax - 10, y + 106, 'which frontier?', size=11, fill=RL_E, anchor='end')
text(cx_ + 10, y + 92, 'CRITIC', size=11.5, bold=True, fill=RL_E)
text(cx_ + 10, y + 106, 'how good is this state?', size=11, fill=RL_E)
rows = [y + 116, y + 170, y + 224, y + 278]
actor = [('128', 'tanh'), ('128', 'tanh'), ('12 scores', 'one per slot'), ('mask + softmax', 'invalid slots → 0')]
critic = [('128', 'tanh'), ('128', 'tanh'), ('V(s)', '1 number')]
for col, items in ((ax, actor), (cx_, critic)):
    for i, (t1, t2) in enumerate(items):
        box(col - 70, rows[i], 140, 34, t1, t2)
        if i:
            path(f'M{col},{rows[i-1]+34} L{col},{rows[i]}')
path(f'M{ax},{rows[3]+34} L{ax},{rows[3]+48}')
text(ax, rows[3] + 66, 'π(slot | s)', size=16, bold=True, fill='#222', anchor='middle',
     extra=' font-family="serif" font-style="italic"')
text(ax, rows[3] + 81, 'probability of each slot', size=10.5, fill=GREY, anchor='middle')
path(f'M{cx_},{rows[2]+34} L{cx_},{rows[2]+48}')
text(cx_, rows[2] + 66, 'expected return from here', size=12, bold=True, anchor='middle')
text(cx_, rows[2] + 81, 'the baseline to judge actions against', size=10.5, fill=GREY, anchor='middle')

# the training loop (right)
lx = x + 612
section(lx, y + 56, 'THE TRAINING LOOP (PPO)')
steps = [
    ('Collect experience', 'run π in 8 simulators × 256 decisions = 2,048 samples'),
    ('Score every decision: advantage', 'Â = return actually got − V(s): better or worse than expected?'),
    ('Update the actor (clipped)', 'raise π where Â &gt; 0, lower where Â &lt; 0, at most ±20 % per update'),
    ('Update the critic', 'fit V(s) to the real returns; entropy bonus keeps exploring'),
]
sy0, sh, gap = y + 76, 54, 22
for i, (name, detail) in enumerate(steps):
    sy = sy0 + i * (sh + gap)
    p.append(f'<rect x="{lx+26}" y="{sy}" width="370" height="{sh}" rx="8" fill="{ROS}" stroke="{ROS_E}" stroke-width="1.5"/>')
    p.append(f'<circle cx="{lx+9}" cy="{sy+sh/2}" r="11" fill="{ROS_E}"/>')
    text(lx + 9, sy + sh / 2 + 4.5, str(i + 1), size=12, anchor='middle', bold=True, fill='white')
    text(lx + 38, sy + 21, name, size=12.5, bold=True)
    text(lx + 38, sy + 38, detail, size=10.5, fill='#444')
    if i < len(steps) - 1:
        path(f'M{lx+9},{sy+sh/2+11} L{lx+9},{sy+sh+gap+sh/2-11}', width=1.4)
top, bot = sy0 + sh / 2, sy0 + 3 * (sh + gap) + sh / 2
path(f'M{lx+396},{bot} L{lx+410},{bot} L{lx+410},{top} L{lx+396},{top}', width=1.6)
text(lx + 422, (top + bot) / 2, 'repeat', size=11.5, anchor='middle', bold=True, fill=ROS_E,
     extra=f' transform="rotate(-90 {lx+422} {(top+bot)/2})"')

with open(OUT, 'w') as fh:
    fh.write('\n'.join(p + ['</svg>']))
print('Wrote', os.path.normpath(OUT))
