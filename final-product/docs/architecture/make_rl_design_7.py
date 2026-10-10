"""
Slide graphic (v7): the RL formulation drawn as a textbook-style figure.

  python3 docs/architecture/make_rl_design_7.py   # writes presentation/rl_design_choices_7.svg

The bar chart under the actor uses the shipped policy's real output for the first decision
of an episode on the benchmark arena (8 valid slots; see docs/ultimate_rl_explanation.md §5.4).
Objective: J = J_CLIP - 0.5 * E_VF + 0.01 * H (vf_coef, ent_coef from rl_sim/train.py).
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'presentation', 'rl_design_choices_7.svg')
W, H = 1100, 560
INK, GREY, HAIR = '#1C1C1C', '#6E6E6E', '#D3D3D3'
ACT, ACT_L = '#2B5C8A', '#DCE7F2'        # actor
CRI, CRI_L = '#B5562B', '#F5E3D8'        # critic
ENT = '#5E7D3A'                          # entropy
SANS = "'Open Sans', sans-serif"
SERIF = SANS                             # sans everywhere, including math
MATH = f' font-family="{SERIF}" font-style="italic"'

p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{SANS}" font-size="13">',
     '<defs><marker id="arr" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
     f'<path d="M0,0 L9,3.5 L0,7 z" fill="{INK}"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']


def text(x, y, s, size=13, anchor='start', bold=False, fill=INK, extra=''):
    b = ' font-weight="bold"' if bold else ''
    p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"{b}{extra}>{s}</text>')


def heading(x, y, s, size=18):
    text(x, y, s, size=size, extra=f' font-family="{SERIF}"')


def hline(x1, x2, y):
    p.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{HAIR}" stroke-width="1"/>')


def vline(x, y1, y2):
    p.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{HAIR}" stroke-width="1"/>')


def m(t, size=14, fill=INK):
    return f'<tspan font-family="{SERIF}" font-style="italic" font-size="{size}" fill="{fill}">{t}</tspan>'


def sub(t):
    return f'<tspan dy="5" font-size="12">{t}</tspan><tspan dy="-5"> </tspan>'


# ── top band: observation, action, reward ───────────────────────────────
heading(40, 46, 'Observation')
for i, line in enumerate(['For every nearby frontier: where it is, how far it is,',
                          'how much is still unknown around it, and how close',
                          'it is to the other robot and to that robot’s goal.',
                          'Plus: both robots’ positions, % mapped, elapsed time.']):
    text(40, 70 + i * 17, line, size=12.5, fill=GREY if i == 3 else INK)
vline(400, 26, 124)
heading(424, 46, 'Action')
text(424, 72, 'Pick one of the 12 nearest frontiers.', size=12.5)
text(424, 90, 'Nav2 then drives the robot there.', size=12.5, fill=GREY)
vline(744, 26, 124)
heading(768, 46, 'Reward')
text(768, 90, 'r = 0.1 · ΔA − 0.01 · Δt', size=22, extra=MATH)
hline(30, 1070, 134)

# ── the network (left) ──────────────────────────────────────────────────
heading(40, 168, 'The network')
vline(620, 150, 545)

IN_Y, H1_Y, H2_Y, OUT_Y = 206, 276, 340, 404
BASE = OUT_Y + 66                        # baseline of the π(slot | s) chart
# critic rows: same first layer as the actor, stretched evenly down to the π row
C_H1, C_H2, C_OUT = H1_Y, (H1_Y + BASE - 4) // 2, BASE - 4
mid, ax, cx = 330, 175, 470


def row(cxn, y, n, gap, r, fill, stroke, ellipsis=True):
    """n node x-positions centred on cxn, with an ellipsis in the middle."""
    half = n // 2
    xs = [cxn + (k - half - (0 if k < half else -1)) * gap if ellipsis else cxn + (k - (n - 1) / 2) * gap
          for k in range(n)]
    return xs


def nodes(xs, y, r, fill, stroke, dashed=False):
    for x in xs:
        d = ' stroke-dasharray="2 2"' if dashed else ''
        p.append(f'<circle cx="{x:.1f}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.3"{d}/>')


def wires(xs1, y1, xs2, y2, r1, r2, color, opacity):
    for a in xs1:
        for b in xs2:
            p.append(f'<line x1="{a:.1f}" y1="{y1 + r1}" x2="{b:.1f}" y2="{y2 - r2}" stroke="{color}" '
                     f'stroke-opacity="{opacity}" stroke-width="0.7"/>')


xin = row(mid, IN_Y, 12, 15, 4.5, 'white', INK)
branches = []
for c, col, light, y1, y2 in ((ax, ACT, ACT_L, H1_Y, H2_Y), (cx, CRI, CRI_L, C_H1, C_H2)):
    h1 = row(c, y1, 8, 16, 5, light, col)
    h2 = row(c, y2, 8, 16, 5, light, col)
    wires(xin, IN_Y, h1, y1, 4.5, 5, col, 0.16)
    wires(h1, y1, h2, y2, 5, 5, col, 0.2)
    branches.append((h1, h2, col, light, y1, y2))

# actor output: 12 slot scores, last 4 masked
outs = [ax + (k - 5.5) * 11 for k in range(12)]
wires(branches[0][1], H2_Y, outs, OUT_Y, 5, 4, ACT, 0.12)
nodes(outs[:8], OUT_Y, 4, ACT, ACT)
nodes(outs[8:], OUT_Y, 4, 'white', '#AAAAAA', dashed=True)
# critic output: one value
wires(branches[1][1], C_H2, [cx], C_OUT, 5, 7, CRI, 0.25)
nodes([cx], C_OUT, 7, CRI, CRI)

nodes(xin, IN_Y, 4.5, 'white', INK)
for h1, h2, col, light, y1, y2 in branches:
    nodes(h1, y1, 5, light, col)
    nodes(h2, y2, 5, light, col)
for x0, y in ((mid, IN_Y), (ax, H1_Y), (ax, H2_Y), (cx, C_H1), (cx, C_H2)):
    text(x0, y + 4, '…', size=13, anchor='middle', fill=GREY)

text(mid + 108, IN_Y + 4, 'observation (153 numbers)', size=11.5, fill=GREY, extra=' font-style="italic"')
for ya, yc in ((H1_Y, C_H1), (H2_Y, C_H2)):
    text(ax - 76, ya + 4, 'hidden layer', size=11, anchor='end', fill=GREY, extra=' font-style="italic"')
    text(cx + 76, yc + 4, 'hidden layer', size=11, fill=GREY, extra=' font-style="italic"')
text(ax - 76, OUT_Y + 4, '12 slot scores', size=11, anchor='end', fill=GREY, extra=' font-style="italic"')

# π(slot | s): real output for the first decision of an episode (8 valid slots)
probs = [0.0029, 0.8626, 0.0070, 0.0017, 0.0036, 0.1192, 0.0013, 0.0016]
base = BASE
p.append(f'<line x1="{outs[0]-6}" y1="{base}" x2="{outs[-1]+6}" y2="{base}" stroke="{INK}" stroke-width="0.8"/>')
for k, x in enumerate(outs):
    if k < 8:
        hgt = max(probs[k] * 46, 1.2)
        p.append(f'<rect x="{x-3.5:.1f}" y="{base-hgt:.1f}" width="7" height="{hgt:.1f}" fill="{ACT}"/>')
    else:
        text(x, base - 2, '×', size=10, anchor='middle', fill='#AAAAAA')
text(outs[1] + 6, base - 32, '0.86', size=10, fill=ACT)
text(outs[5] + 6, base - 8, '0.12', size=10, fill=ACT)
text(ax - 76, base - 4, 'mask + softmax', size=11, anchor='end', fill=GREY, extra=' font-style="italic"')
text(outs[-1] + 14, base - 4, 'π(slot | s)', size=16, extra=MATH)

text(cx + 14, C_OUT + 5, 'V(s)', size=16, extra=MATH)
text(cx + 14, C_OUT + 22, 'expected return', size=11, fill=GREY, extra=' font-style="italic"')

text(ax, 522, f'<tspan font-weight="bold" fill="{ACT}">Actor</tspan><tspan dx="5" fill="{GREY}">· which frontier?</tspan>',
     size=12.5, anchor='middle')
text(cx, 522, f'<tspan font-weight="bold" fill="{CRI}">Critic</tspan><tspan dx="5" fill="{GREY}">· how good is this state?</tspan>',
     size=12.5, anchor='middle')

# ── training loop (right) ───────────────────────────────────────────────
heading(644, 168, 'Training loop (PPO)')
steps = [
    ('Collect experience', f'run {m("π")} in 8 simulators for 256 decisions'),
    ('Score every decision (advantage)', m('Â = R − V(s)')),
    ('Update the actor (clipped)', f'raise {m("π")} where {m("Â &gt; 0")}, lower where {m("Â &lt; 0")}'),
    ('Update the critic', f'fit {m("V(s)")} to the real return {m("R")}'),
]
for i, (name, detail) in enumerate(steps):
    y = 206 + i * 50
    text(650, y + 6, str(i + 1), size=26, fill=ACT, extra=f' font-family="{SERIF}"')
    text(684, y - 4, name, size=13.5, bold=True)
    text(684, y + 14, detail, size=12, fill=GREY)
# return arrow, step 4 back to step 1: a straight bracket with rounded corners
p.append(f'<path d="M914,362 H920 Q932,362 932,350 V210 Q932,198 920,198 H904" fill="none" '
         f'stroke="{GREY}" stroke-width="1.3" marker-end="url(#arr)"/>')
text(950, 280, 'repeat', size=12, anchor='middle', fill=GREY,
     extra=' font-style="italic" transform="rotate(-90 950 280)"')

# ── objective ───────────────────────────────────────────────────────────
hline(644, 1060, 400)
heading(644, 428, 'Objective', size=16)
ey = 466
text(650, ey, 'J(θ) =', size=20, extra=MATH)
terms = [(762, 'J' + sub('CLIP'), ACT, 726, 798, 'improve the actor'),
         (880, '0.5 · E' + sub('VF'), CRI, 838, 922, 'fix the critic’s error'),
         (1004, '0.01 · H(π)', ENT, 958, 1052, 'keep exploring')]
for tx, t, col, x1, x2, lab in terms:
    text(tx, ey, t, size=20, fill=col, anchor='middle', extra=MATH)
    yb, mx = ey + 10, (x1 + x2) / 2
    p.append(f'<path d="M{x1},{yb} q0,7 7,7 L{mx-7},{yb+7} q7,0 7,7 q0,-7 7,-7 L{x2-7},{yb+7} q7,0 7,-7" '
             f'fill="none" stroke="{col}" stroke-width="1.2"/>')
    text(mx, yb + 30, lab, size=11.5, fill=col, anchor='middle')
text(814, ey, '−', size=20, anchor='middle')
text(940, ey, '+', size=20, anchor='middle')

with open(OUT, 'w') as fh:
    fh.write('\n'.join(p + ['</svg>']))
print('Wrote', os.path.normpath(OUT))
