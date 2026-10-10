"""
Horizontal strip for "The 2D training simulator" slide (under the GIF): simulator speed + tech stack,
in the same design language as rl_design_choices_7.

  python3 docs/architecture/make_sim_strip.py   # writes presentation/rl_simulator_strip.svg

Throughput numbers: see make_rl_design_2.py.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'presentation', 'rl_simulator_strip.svg')
W, H = 1100, 190
INK, GREY, HAIR = '#1C1C1C', '#6E6E6E', '#D3D3D3'
ACT, CRI = '#2B5C8A', '#B5562B'
SANS = "'Open Sans', sans-serif"

p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{SANS}" font-size="13">',
     f'<rect width="{W}" height="{H}" fill="white"/>']


def text(x, y, s, size=13, anchor='start', bold=False, fill=INK, extra=''):
    b = ' font-weight="bold"' if bold else ''
    p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"{b}{extra}>{s}</text>')


# ── left: fast 2D simulator ─────────────────────────────────────────────
text(30, 38, 'Train in a fast 2D simulator', size=18)
text(30, 60, 'same arena, frontiers and speed as Gazebo', size=12.5, fill=GREY)
for x, val, cap1, cap2, col in [(30, '≈ 190', 'decisions / s', '2D simulator', ACT),
                                (200, '≈ 0.7', 'decisions / s', 'Gazebo, 8 headless', INK),
                                (370, '≈ 300×', 'more decisions / s', '2D vs Gazebo', ACT)]:
    text(x, 116, val, size=34, fill=col)
    text(x, 140, cap1, size=12, fill=GREY)
    text(x, 157, cap2, size=12, bold=True, fill=col)
p.append(f'<line x1="545" y1="22" x2="545" y2="{H-22}" stroke="{HAIR}" stroke-width="1"/>')

# ── right: tech stack ───────────────────────────────────────────────────
text(575, 38, 'Tech stack', size=18)
rows = [('RL training', ACT, ['Stable-Baselines3 · MaskablePPO · PyTorch', 'Gymnasium · TensorBoard']),
        ('2D simulator', CRI, ['NumPy · OpenCV'])]
y = 82
for label, col, lines in rows:
    text(575, y, label, size=13, bold=True, fill=col)
    for i, line in enumerate(lines):
        text(700, y + i * 20, line, size=13)
    y += len(lines) * 20 + 22

with open(OUT, 'w') as fh:
    fh.write('\n'.join(p + ['</svg>']))
print('Wrote', os.path.normpath(OUT))
