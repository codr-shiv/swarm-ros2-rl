"""
Slide graphic: four design choices of the RL formulation, one panel each.

  python3 docs/architecture/make_rl_design.py   # writes presentation/rl_design_choices.svg
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'presentation', 'rl_design_choices.svg')
W, H = 1100, 560
PW, PH = 515, 225
RL, RL_E = '#F3E8FA', '#7B3FA6'
SIM, SIM_E = '#E8F1FB', '#1F5FA8'
GREY = '#666'

p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="13">',
     '<defs><marker id="arr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
     '<path d="M0,0 L10,4 L0,8 z" fill="#333"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="32" text-anchor="middle" font-size="20" font-weight="bold">'
     'RL brain: why the problem is easy to learn</text>']


def panel(x, y, title, sub):
    p.append(f'<rect x="{x}" y="{y}" width="{PW}" height="{PH}" rx="12" fill="#FAFAFA" stroke="#DDD" stroke-width="1.5"/>')
    p.append(f'<text x="{x+18}" y="{y+30}" font-size="16" font-weight="bold">{title}</text>')
    p.append(f'<text x="{x+18}" y="{y+50}" font-size="12.5" fill="{GREY}">{sub}</text>')


def text(x, y, s, size=12.5, anchor='start', bold=False, fill='#333'):
    b = ' font-weight="bold"' if bold else ''
    p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"{b}>{s}</text>')


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

# 2. Feature vector, not an image ───────────────────────────────────────
x, y = 555, 60
panel(x, y, '2 · 153 numbers, not a map image', 'compact, ego-centric features learn faster')
gx, gy, c = x + 30, y + 66, 6                          # map image: 20×20 drawn, stands for 160×160
for r in range(20):
    for k in range(20):
        shade = '#CFCFCF' if (r * 7 + k * 3) % 11 < 3 else ('#9A9A9A' if (r + k) % 13 == 0 else '#EDEDED')
        p.append(f'<rect x="{gx+k*c}" y="{gy+r*c}" width="{c-1}" height="{c-1}" fill="{shade}"/>')
text(gx + 60, gy + 138, 'map image: 25,600 cells', anchor='middle', fill=GREY)
p.append(f'<path d="M{gx+140},{gy+60} L{gx+190},{gy+60}" stroke="#333" stroke-width="1.8" marker-end="url(#arr)"/>')
vx, vy, s = x + 250, y + 64, 8                         # 12 candidates × 12 features + 9 global
for r in range(12):
    for k in range(12):
        p.append(f'<rect x="{vx+k*s}" y="{vy+r*s}" width="{s-1.5}" height="{s-1.5}" fill="{RL}" stroke="{RL_E}" stroke-width="0.6"/>')
for k in range(9):
    p.append(f'<rect x="{vx+k*s}" y="{vy+12*s+5}" width="{s-1.5}" height="{s-1.5}" fill="{SIM}" stroke="{SIM_E}" stroke-width="0.6"/>')
text(vx + 108, vy + 38, '12 candidates', fill=RL_E, bold=True)
text(vx + 108, vy + 54, '× 12 features', fill=RL_E)
text(vx + 108, vy + 109, '+ 9 global', fill=SIM_E, bold=True)
text(vx + 48, vy + 138, '= 153', anchor='middle', bold=True, size=14)

# 3. Fast 2D simulator ──────────────────────────────────────────────────
x, y = 30, 315
panel(x, y, '3 · Train in a fast 2D simulator', 'time for one episode (bars to scale)')
text(x + 24, y + 92, 'Gazebo', bold=True)
p.append(f'<rect x="{x+100}" y="{y+77}" width="300" height="22" rx="4" fill="{SIM}" stroke="{SIM_E}" stroke-width="1.5"/>')
text(x + 410, y + 93, '≈ 5 min')
text(x + 24, y + 137, '2D sim', bold=True)
p.append(f'<rect x="{x+100}" y="{y+122}" width="3" height="22" fill="{RL_E}"/>')
text(x + 112, y + 138, '≈ 0.1 s')
text(x + PW / 2, y + 195, '≈ 3000× faster  →  300k decisions in ~25 min on a laptop', anchor='middle',
     bold=True, size=14, fill=RL_E)

# 4. Network + PPO ──────────────────────────────────────────────────────
x, y = 555, 315
panel(x, y, '4 · Small network, trained with PPO', 'MaskablePPO · 8 parallel environments')


def box(bx, by, bw, bh, t1, t2, fill=RL, edge=RL_E):
    p.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" fill="{fill}" stroke="{edge}" stroke-width="1.6"/>')
    text(bx + bw / 2, by + bh / 2 - 2, t1, anchor='middle', bold=True)
    text(bx + bw / 2, by + bh / 2 + 14, t2, anchor='middle', size=11, fill=GREY)


def arrow(x1, y1, x2, y2):
    p.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="#333" stroke-width="1.6" marker-end="url(#arr)"/>')


cy = y + 135
box(x + 18, cy - 25, 78, 50, 'obs', '153', SIM, SIM_E)
box(x + 122, cy - 45, 62, 90, '128', 'hidden')
box(x + 206, cy - 45, 62, 90, '128', 'hidden')
box(x + 310, cy - 70, 180, 50, 'Actor', 'score for each of 12 slots')
box(x + 310, cy + 20, 180, 50, 'Critic', 'expected return')
arrow(x + 96, cy, x + 122, cy); arrow(x + 184, cy, x + 206, cy)
arrow(x + 268, cy - 10, x + 310, cy - 45); arrow(x + 268, cy + 10, x + 310, cy + 45)
text(x + 195, y + 203, 'actor and critic: 2 × 128 MLP each', anchor='middle', size=11.5, fill=GREY)

with open(OUT, 'w') as fh:
    fh.write('\n'.join(p + ['</svg>']))
print('Wrote', os.path.normpath(OUT))
