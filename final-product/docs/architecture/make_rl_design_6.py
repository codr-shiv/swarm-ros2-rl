"""
Slide graphic (v6): observation/action/reward strip, the actor-critic network, the PPO loop
and the PPO objective. No titles; the slide supplies its own.

  python3 docs/architecture/make_rl_design_6.py   # writes presentation/rl_design_choices_6.svg

Objective (SB3 minimises its negative): J = J_CLIP - vf_coef * E_VF + ent_coef * H,
with vf_coef = 0.5 and ent_coef = 0.01 from rl_sim/train.py.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'presentation', 'rl_design_choices_6.svg')
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


MATH = ' font-family="serif" font-style="italic"'


def sub(s):
    """Subscript inside an SVG <text>."""
    return f'<tspan dy="5" font-size="12">{s}</tspan><tspan dy="-5"> </tspan>'


# Observation, action, reward (top strip) ───────────────────────────────
x, y, w = 30, 16, 1040
p.append(f'<rect x="{x}" y="{y}" width="{w}" height="92" rx="12" fill="#FAFAFA" stroke="#DDD" stroke-width="1.5"/>')
cols = [x + w / 6, x + w / 2, x + 5 * w / 6]
tag(cols[0] - 66, y + 14, 'OBSERVATION', SIM, SIM_E, w=132)
text(cols[0], y + 58, '153 numbers per decision', size=13, bold=True, anchor='middle')
text(cols[0], y + 75, '12 frontiers × 12 features + 9 global', size=13, bold=True, anchor='middle')
tag(cols[1] - 66, y + 14, 'ACTION', RL, RL_E, w=132)
text(cols[1], y + 58, 'pick 1 of the 12', size=13, bold=True, anchor='middle')
text(cols[1], y + 75, 'nearest frontiers', size=13, bold=True, anchor='middle')
tag(cols[2] - 66, y + 14, 'REWARD', NAV, NAV_E, w=132)
text(cols[2], y + 70, 'r = 0.1 · ΔA − 0.01 · Δt', size=19, bold=True, fill='#222', anchor='middle', extra=MATH)
for dx in (w / 3, 2 * w / 3):
    p.append(f'<line x1="{x+dx}" y1="{y+14}" x2="{x+dx}" y2="{y+78}" stroke="#E2E2E2" stroke-width="1.5"/>')

# Main panel ────────────────────────────────────────────────────────────
x, y, w, h = 30, 122, 1040, 424
p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#FAFAFA" stroke="#DDD" stroke-width="1.5"/>')
p.append(f'<line x1="{x+590}" y1="{y+20}" x2="{x+590}" y2="{y+h-20}" stroke="#E2E2E2" stroke-width="1.5"/>')

# the network (left): one input, two separate networks on tinted backdrops
section(x + 295, y + 30, 'THE NETWORK')
p[-1] = p[-1].replace('text-anchor="start"', 'text-anchor="middle"')
ax, cx_ = x + 165, x + 425
box(x + 235, y + 45, 120, 44, '153', 'observation', SIM, SIM_E)
top = y + 96
for col, name, q in ((ax, 'ACTOR', 'which frontier?'), (cx_, 'CRITIC', 'how good is this state?')):
    p.append(f'<rect x="{col-110}" y="{top}" width="220" height="{y+412-top}" rx="12" fill="#F7F0FB" '
             f'stroke="#D9C4E8" stroke-width="1.4"/>')
    text(col, top + 20, name, size=12, bold=True, fill=RL_E, anchor='middle', extra=' letter-spacing="1"')
    text(col, top + 34, q, size=11, fill=RL_E, anchor='middle')
path(f'M{x+235},{y+67} L{ax},{y+67} L{ax},{top}')
path(f'M{x+355},{y+67} L{cx_},{y+67} L{cx_},{top}')
rows = [y + 142, y + 202, y + 262, y + 322]
actor = [('Hidden layer 1', '128 neurons'), ('Hidden layer 2', '128 neurons'),
         ('12 scores', 'one per slot'), ('mask + softmax', 'invalid slots → 0')]
critic = [('Hidden layer 1', '128 neurons'), ('Hidden layer 2', '128 neurons'), ('V(s)', '1 number')]
for col, items in ((ax, actor), (cx_, critic)):
    for i, (t1, t2) in enumerate(items):
        box(col - 80, rows[i], 160, 36, t1, t2)
        if i:
            path(f'M{col},{rows[i-1]+36} L{col},{rows[i]}')
path(f'M{ax},{rows[3]+36} L{ax},{rows[3]+48}')
text(ax, rows[3] + 68, 'π(slot | s)', size=16, bold=True, fill='#222', anchor='middle', extra=MATH)
text(ax, rows[3] + 82, 'probability of each slot', size=10.5, fill=GREY, anchor='middle')
path(f'M{cx_},{rows[2]+36} L{cx_},{rows[2]+48}')
text(cx_, rows[2] + 66, 'expected return from here', size=12, bold=True, anchor='middle')
text(cx_, rows[2] + 81, 'the baseline to judge actions against', size=10.5, fill=GREY, anchor='middle')

# the training loop (right)
lx = x + 612
section(lx + 205, y + 30, 'THE TRAINING LOOP (PPO)')
p[-1] = p[-1].replace('text-anchor="start"', 'text-anchor="middle"')
def m(t):
    """Inline math inside a detail line."""
    return f'<tspan font-family="serif" font-style="italic" font-size="13" fill="#222">{t}</tspan>'


steps = [
    ('Collect experience', f'run {m("π")} in 8 simulators for 256 decisions'),
    ('Score every decision (advantage)',
     m("Â = R − V(s)")),
    ('Update the actor (clipped)', f'raise {m("π")} where {m("Â &gt; 0")}, lower where {m("Â &lt; 0")}'),
    ('Update the critic', f'fit {m("V(s)")} to the real return {m("R")}'),
]
sy0, sh, gap = y + 46, 46, 14
for i, (name, detail) in enumerate(steps):
    sy = sy0 + i * (sh + gap)
    p.append(f'<rect x="{lx+26}" y="{sy}" width="370" height="{sh}" rx="8" fill="{ROS}" stroke="{ROS_E}" stroke-width="1.5"/>')
    p.append(f'<circle cx="{lx+9}" cy="{sy+sh/2}" r="11" fill="{ROS_E}"/>')
    text(lx + 9, sy + sh / 2 + 4.5, str(i + 1), size=12, anchor='middle', bold=True, fill='white')
    text(lx + 38, sy + 19, name, size=12.5, bold=True)
    text(lx + 38, sy + 36, detail, size=11, fill='#555')
    if i < len(steps) - 1:
        path(f'M{lx+9},{sy+sh/2+11} L{lx+9},{sy+sh+gap+sh/2-11}', width=1.4)
t_, b_ = sy0 + sh / 2, sy0 + 3 * (sh + gap) + sh / 2
path(f'M{lx+396},{b_} L{lx+410},{b_} L{lx+410},{t_} L{lx+396},{t_}', width=1.6)
text(lx + 422, (t_ + b_) / 2, 'repeat', size=11.5, anchor='middle', bold=True, fill=ROS_E,
     extra=f' transform="rotate(-90 {lx+422} {(t_+b_)/2})"')

# the update rule (right, below the loop)
cl, ct, cw, ch = lx, y + 290, 410, 122
p.append(f'<rect x="{cl}" y="{ct}" width="{cw}" height="{ch}" rx="10" fill="white" stroke="#CCC" stroke-width="1.5"/>')
text(cl + cw / 2, ct + 20, 'THE OBJECTIVE: maximised on every minibatch', size=10.5, bold=True, fill='#999',
     anchor='middle', extra=' letter-spacing="0.5"')
ey = ct + 56
text(cl + 10, ey, 'J(θ) =', size=18, bold=True, fill='#222', extra=MATH)
terms = [(cl + 112, 'J' + sub('CLIP'), RL_E, 'improve the actor', 'min(ρÂ, clip(ρ)·Â)'),
         (cl + 232, '0.5 · E' + sub('VF'), SIM_E, 'fix the critic’s errors', '(V(s) − R)²'),
         (cl + 350, '0.01 · H(π)', NAV_E, 'entropy bonus', 'keep exploring')]
for tx, t, col, label, formula in terms:
    text(tx, ey, t, size=17, bold=True, fill=col, anchor='middle', extra=MATH)
    p.append(f'<rect x="{tx-46}" y="{ey+9}" width="92" height="3" rx="1.5" fill="{col}"/>')
    text(tx, ey + 30, label, size=11, bold=True, fill=col, anchor='middle')
    text(tx, ey + 46, formula, size=11.5, fill='#555', anchor='middle', extra=MATH)
text(cl + 172, ey, '−', size=20, bold=True, fill='#222', anchor='middle')
text(cl + 291, ey, '+', size=20, bold=True, fill='#222', anchor='middle')

with open(OUT, 'w') as fh:
    fh.write('\n'.join(p + ['</svg>']))
print('Wrote', os.path.normpath(OUT))
