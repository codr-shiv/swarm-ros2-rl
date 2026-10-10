"""
Side panel for the "PPO maps faster" slide: how ros_policy_node bridges Gazebo/ROS 2 and the
trained policy. Same design language as rl_design_choices_7.

  python3 docs/architecture/make_policy_node_panel.py   # writes presentation/policy_node_panel.svg
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'presentation', 'policy_node_panel.svg')
W, H = 470, 326
INK, GREY, HAIR = '#1C1C1C', '#6E6E6E', '#D3D3D3'
ACT = '#2B5C8A'
SANS = "'Open Sans', sans-serif"

p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{SANS}" font-size="13">',
     '<defs><marker id="arr" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
     f'<path d="M0,0 L9,3.5 L0,7 z" fill="{GREY}"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']


def text(x, y, s, size=13, anchor='start', bold=False, fill=INK, extra=''):
    b = ' font-weight="bold"' if bold else ''
    p.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"{b}{extra}>{s}</text>')


text(W / 2, 32, 'ros_policy_node', size=19, anchor='middle')

stages = [
    ('Read the live ROS 2 state', 'merged /map · robot poses (TF) · Nav2 goal status', INK),
    ('Prepare the observation array', '', ACT),
    ('Get the best frontier choice using trained policy', '', ACT),
    ('Send the goal', 'NavigateToPose action on /robotN/navigate_to_pose', INK),
]
X0, Y0, STEP = 62, 80, 66
ys = [Y0 + i * STEP for i in range(len(stages))]
p.append(f'<line x1="{X0}" y1="{ys[0]}" x2="{X0}" y2="{ys[-1]}" stroke="{HAIR}" stroke-width="2"/>')
for y, (title, detail, col) in zip(ys, stages):
    p.append(f'<circle cx="{X0}" cy="{y}" r="6" fill="{col}"/>')
    text(X0 + 20, y + 5, title, size=14, bold=True, fill=col)
    text(X0 + 20, y + 25, detail, size=12, fill=GREY)
# loop: back from the last stage to the first, once per second
p.append(f'<path d="M{X0-10},{ys[-1]} H{X0-22} Q{X0-30},{ys[-1]} {X0-30},{ys[-1]-8} V{ys[0]+8} '
         f'Q{X0-30},{ys[0]} {X0-22},{ys[0]} H{X0-11}" fill="none" stroke="{GREY}" stroke-width="1.3" '
         f'marker-end="url(#arr)"/>')
mid = (ys[0] + ys[-1]) / 2
text(X0 - 36, mid, 'every 1 s', size=11.5, anchor='middle', fill=GREY,
     extra=f' font-style="italic" transform="rotate(-90 {X0-36} {mid})"')


with open(OUT, 'w') as fh:
    fh.write('\n'.join(p + ['</svg>']))
print('Wrote', os.path.normpath(OUT))
