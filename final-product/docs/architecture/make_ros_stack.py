"""
Small "ROS 2 tech stack" card, same style as the simulator sidebar.

  python3 docs/architecture/make_ros_stack.py   # writes presentation/ros_tech_stack.svg
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'presentation', 'ros_tech_stack.svg')
W, H = 380, 124
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


X, Y, CW, CH = 6, 6, 368, 112
p.append(f'<rect x="{X}" y="{Y}" width="{CW}" height="{CH}" rx="10" fill="#FAFAFA" stroke="#DDD" stroke-width="1.5"/>')
text(X + 14, Y + 24, 'Tech stack', size=14.5, bold=True)
tag(X + 14, Y + 40, 'ROS 2', SIM, SIM_E, w=112)
for i, line in enumerate(['ROS 2 Humble · rclpy · TF2', 'Gazebo Classic 11 · TurtleBot3',
                          'slam_toolbox · Nav2']):
    text(X + 138, Y + 55 + i * 16, line, size=11.5, bold=True)

with open(OUT, 'w') as fh:
    fh.write('\n'.join(p + ['</svg>']))
print('Wrote', os.path.normpath(OUT))
