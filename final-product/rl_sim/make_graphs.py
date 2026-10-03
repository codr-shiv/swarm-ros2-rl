"""
Generate every comparison chart (SVG) into graphs/ from the data in results/.

  cd ~/swarm/final-product && python3 rl_sim/make_graphs.py

Inputs:  results/training/eval_history.csv   (written by train.py)
         results/training/tensorboard_scalars.csv (written by train.py / export_tensorboard.py)
         results/gazebo_runs/*.csv            (written by gazebo_test.sh)
         models/ppo_frontier_policy.zip       (for the 2D simulator comparison)
Outputs: graphs/*.svg and results/sim2d/policy_comparison.csv
"""
import argparse
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from rl_sim.analyze_gazebo_tests import CHECKPOINTS, load, metrics, svg as coverage_svg  # noqa: E402

PPO_C, HEUR_C, OTHER_C = '#1f77b4', '#d62728', '#7f7f7f'
W, H, L, R, T, B = 640, 400, 70, 20, 50, 60


# ── tiny SVG toolkit ─────────────────────────────────────────────────────
class Chart:
    def __init__(self, title, ylabel, ymax, ymin=0.0, xlabel=''):
        self.ymin, self.ymax = ymin, ymax
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                      f'font-family="sans-serif" font-size="12">',
                      f'<rect width="{W}" height="{H}" fill="white"/>',
                      f'<text x="{W/2}" y="24" text-anchor="middle" font-size="15" font-weight="bold">{title}</text>',
                      f'<text x="18" y="{(T+H-B)/2}" transform="rotate(-90 18 {(T+H-B)/2})" '
                      f'text-anchor="middle">{ylabel}</text>']
        if xlabel:
            self.parts.append(f'<text x="{(L+W-R)/2}" y="{H-14}" text-anchor="middle">{xlabel}</text>')
        step = nice_step(ymax - ymin)
        v = np.ceil(ymin / step) * step
        while v <= ymax + 1e-9:
            y = self.y(v)
            self.parts.append(f'<line x1="{L}" x2="{W-R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e5e5"/>'
                              f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end">{fmt(v)}</text>')
            v += step
        self.parts.append(f'<line x1="{L}" x2="{W-R}" y1="{self.y(max(ymin, 0)):.1f}" '
                          f'y2="{self.y(max(ymin, 0)):.1f}" stroke="#333"/>')

    def y(self, v):
        return H - B - (H - B - T) * (v - self.ymin) / (self.ymax - self.ymin)

    def add(self, s):
        self.parts.append(s)

    def legend(self, items, x=None, y=None):
        x = x or L + 12
        y = y or T + 8
        for k, (label, color) in enumerate(items):
            yy = y + 18 * k
            self.add(f'<rect x="{x}" y="{yy-9}" width="14" height="10" fill="{color}"/>'
                     f'<text x="{x+20}" y="{yy}">{label}</text>')

    def save(self, path):
        with open(path, 'w') as fh:
            fh.write('\n'.join(self.parts + ['</svg>']))


def nice_step(span):
    raw = span / 6
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def fmt(v):
    if abs(v) < 1e-12:
        v = 0.0
    return f'{v:.0f}' if abs(v - round(v)) < 1e-9 else f'{v:.2f}'.rstrip('0').rstrip('.')


def bars(path, title, ylabel, groups, series, values, errors=None, dots=None, ymin=0.0, note=''):
    """Grouped bar chart. values[s][g], errors[s][g], dots[s][g] = list of points."""
    top = max(max(v + (errors[s][g] if errors else 0) for g, v in enumerate(vals))
              for s, vals in enumerate(values))
    if dots:
        top = max(top, max(max(d) for ds in dots for d in ds))
    c = Chart(title, ylabel, top * 1.15, ymin)
    n_g, n_s = len(groups), len(series)
    gw = (W - L - R) / n_g
    bw = gw * 0.7 / n_s
    for g, name in enumerate(groups):
        x0 = L + g * gw + gw * 0.15
        c.add(f'<text x="{L + g*gw + gw/2:.1f}" y="{H-B+18}" text-anchor="middle">{name}</text>')
        for s, (label, color) in enumerate(series):
            v = values[s][g]
            x = x0 + s * bw
            y0, y1 = c.y(max(ymin, 0)), c.y(v)
            c.add(f'<rect x="{x:.1f}" y="{min(y0, y1):.1f}" width="{bw*0.92:.1f}" height="{abs(y0-y1):.1f}" '
                  f'fill="{color}" fill-opacity="0.85"/>')
            c.add(f'<text x="{x + bw*0.46:.1f}" y="{min(y0, y1) - 4 - (0 if not errors else 0):.1f}" '
                  f'text-anchor="middle" font-size="11">{v:.1f}</text>' if not errors else '')
            if errors:
                e = errors[s][g]
                cx = x + bw * 0.46
                c.add(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{c.y(v-e):.1f}" y2="{c.y(v+e):.1f}" stroke="#222"/>'
                      f'<line x1="{cx-5:.1f}" x2="{cx+5:.1f}" y1="{c.y(v+e):.1f}" y2="{c.y(v+e):.1f}" stroke="#222"/>'
                      f'<line x1="{cx-5:.1f}" x2="{cx+5:.1f}" y1="{c.y(v-e):.1f}" y2="{c.y(v-e):.1f}" stroke="#222"/>')
            if dots:
                for k, d in enumerate(dots[s][g]):
                    jitter = (k - (len(dots[s][g]) - 1) / 2) * bw * 0.08
                    c.add(f'<circle cx="{x + bw*0.46 + jitter:.1f}" cy="{c.y(d):.1f}" r="2.6" '
                          f'fill="white" stroke="#222"/>')
    c.legend(series)
    if note:
        c.add(f'<text x="{W-R}" y="{T-6}" text-anchor="end" font-size="11" fill="#555">{note}</text>')
    c.save(path)


# ── training-metric line panels (from TensorBoard scalars) ───────────────
TRAINING_METRICS = [
    # (tag, title, y label, transform, note)
    ('rollout/ep_rew_mean', 'Mean episode reward during training', 'reward', None, 'higher is better'),
    ('rollout/ep_len_mean', 'Mean episode length during training', 'decisions per episode', None, 'fewer = faster exploration'),
    ('train/entropy_loss', 'Policy entropy', 'entropy (nats)', lambda v: -v, 'falls as the policy becomes confident'),
    ('train/value_loss', 'Value-function loss', 'loss', None, ''),
    ('train/policy_gradient_loss', 'Policy-gradient loss', 'loss', None, ''),
    ('train/explained_variance', 'Explained variance of the value function', 'explained variance', None, '1 = perfect value prediction'),
    ('train/approx_kl', 'Approximate KL divergence per update', 'KL', None, 'size of each policy update'),
    ('train/clip_fraction', 'PPO clip fraction', 'fraction clipped', None, ''),
    ('train/learning_rate', 'Learning rate (linear decay)', 'learning rate', None, ''),
]


def line_panel(parts, x0, y0, w, h, steps, values, title, ylabel, note='', ref=None, small=False):
    """Draw one line chart into *parts* inside the box (x0, y0, w, h)."""
    fs = 10 if small else 12
    l, r, t, b = (48 if small else 70), 10, (34 if small else 50), (30 if small else 60)
    vals = np.asarray(values, float)
    lo, hi = float(vals.min()), float(vals.max())
    if ref is not None:
        lo, hi = min(lo, ref), max(hi, ref)
    pad = (hi - lo) * 0.1 or abs(hi) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    xmax = float(max(steps)) or 1.0

    def X(v):
        return x0 + l + (w - l - r) * v / xmax

    def Y(v):
        return y0 + h - b - (h - b - t) * (v - lo) / (hi - lo)
    parts.append(f'<text x="{x0 + w/2}" y="{y0 + (16 if small else 24)}" text-anchor="middle" '
                 f'font-size="{fs + 2}" font-weight="bold">{title}</text>')
    if note:
        parts.append(f'<text x="{x0 + w - r}" y="{y0 + t - 6}" text-anchor="end" font-size="{fs - 1}" '
                     f'fill="#555">{note}</text>')
    step = nice_step(hi - lo)
    v = np.ceil(lo / step) * step
    while v <= hi + 1e-12:
        label = f'{v:.0e}' if 1e-12 < abs(v) < 1e-3 else fmt(round(v, 6))
        parts.append(f'<line x1="{x0+l}" x2="{x0+w-r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e5e5e5"/>'
                     f'<text x="{x0+l-6}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="{fs}">{label}</text>')
        v += step
    xstep = nice_step(xmax) if small else nice_step(xmax) / 2
    for k in np.arange(0, xmax + 1, xstep):
        parts.append(f'<text x="{X(k):.1f}" y="{y0+h-b+(14 if small else 18)}" text-anchor="middle" '
                     f'font-size="{fs}">{k/1000:.0f}k</text>')
    parts.append(f'<text x="{x0 + (l+w-r)/2}" y="{y0+h-(4 if small else 14)}" text-anchor="middle" '
                 f'font-size="{fs}">training decisions</text>')
    parts.append(f'<text x="{x0+12}" y="{y0+(t+h-b)/2}" transform="rotate(-90 {x0+12} {y0+(t+h-b)/2})" '
                 f'text-anchor="middle" font-size="{fs}">{ylabel}</text>')
    if ref is not None:
        parts.append(f'<line x1="{x0+l}" x2="{x0+w-r}" y1="{Y(ref):.1f}" y2="{Y(ref):.1f}" stroke="{HEUR_C}" '
                     f'stroke-width="1.8" stroke-dasharray="6 4"/>'
                     f'<text x="{x0+w-r-4}" y="{Y(ref)-5:.1f}" text-anchor="end" font-size="{fs}" '
                     f'fill="{HEUR_C}">heuristic</text>')
    pts = ' '.join(f'{X(a):.1f},{Y(c):.1f}' for a, c in zip(steps, vals))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{PPO_C}" stroke-width="{1.8 if small else 2.2}"/>')


def training_metric_graphs(out, scalars_csv, heuristic_return):
    data = {}
    for row in csv.DictReader(open(scalars_csv)):
        data.setdefault(row['tag'], []).append((float(row['step']), float(row['value'])))
    panels = []
    for k, (tag, title, ylabel, fn, note) in enumerate(TRAINING_METRICS, start=11):
        if tag not in data:
            continue
        pts = sorted(data[tag])
        steps = [a for a, _ in pts]
        vals = [fn(b) if fn else b for _, b in pts]
        ref = heuristic_return if tag == 'rollout/ep_rew_mean' else None
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">',
                 f'<rect width="{W}" height="{H}" fill="white"/>']
        line_panel(parts, 0, 0, W, H, steps, vals, title, ylabel, note, ref)
        slug = tag.split('/')[1]
        with open(os.path.join(out, f'{k:02d}_training_{slug}.svg'), 'w') as fh:
            fh.write('\n'.join(parts + ['</svg>']))
        panels.append((steps, vals, title, ylabel, ref))
    # 3x3 dashboard of all training metrics
    pw, ph = 420, 280
    cols = 3
    rows = (len(panels) + cols - 1) // cols
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{pw*cols}" height="{ph*rows + 40}" '
             f'font-family="sans-serif">', f'<rect width="{pw*cols}" height="{ph*rows + 40}" fill="white"/>',
             f'<text x="{pw*cols/2}" y="26" text-anchor="middle" font-size="18" font-weight="bold">'
             f'PPO training curves (2D simulator)</text>']
    for i, (steps, vals, title, ylabel, ref) in enumerate(panels):
        line_panel(parts, (i % cols) * pw, 40 + (i // cols) * ph, pw, ph, steps, vals, title, ylabel,
                   ref=ref, small=True)
    with open(os.path.join(out, '20_training_dashboard.svg'), 'w') as fh:
        fh.write('\n'.join(parts + ['</svg>']))


# ── individual graphs ────────────────────────────────────────────────────
def training_curve(path_ret, path_time, hist):
    x = np.array([float(r['decisions']) for r in hist])
    for path, key, hkey, title, ylabel, better in (
            (path_ret, 'return', 'heuristic_return', 'Training progress: episode return (2D simulator)',
             'deterministic episode return', 'higher is better'),
            (path_time, 'explore_time_s', 'heuristic_explore_time_s',
             'Training progress: time to explore the arena (2D simulator)', 'time to full exploration (s)',
             'lower is better')):
        y = np.array([float(r[key]) for r in hist])
        h = float(hist[0][hkey])
        lo, hi = min(y.min(), h), max(y.max(), h)
        pad = (hi - lo) * 0.15 or 1
        c = Chart(title, ylabel, hi + pad, max(lo - pad, 0), 'training decisions')
        xmax = x.max()

        def xs(v):
            return L + (W - L - R) * v / xmax
        for v in np.linspace(0, xmax, 7):
            c.add(f'<text x="{xs(v):.1f}" y="{H-B+18}" text-anchor="middle">{v/1000:.0f}k</text>')
        c.add(f'<line x1="{L}" x2="{W-R}" y1="{c.y(h):.1f}" y2="{c.y(h):.1f}" stroke="{HEUR_C}" '
              f'stroke-width="2" stroke-dasharray="6 4"/>')
        pts = ' '.join(f'{xs(a):.1f},{c.y(b):.1f}' for a, b in zip(x, y))
        c.add(f'<polyline points="{pts}" fill="none" stroke="{PPO_C}" stroke-width="2.5"/>')
        for a, b in zip(x, y):
            c.add(f'<circle cx="{xs(a):.1f}" cy="{c.y(b):.1f}" r="3" fill="{PPO_C}"/>')
        c.legend([('PPO policy (evaluated every 10k decisions)', PPO_C),
                  ('Heuristic (frontier_coordinator)', HEUR_C)], x=W - 330, y=T + 8)
        c.add(f'<text x="{W-R}" y="{T-6}" text-anchor="end" font-size="11" fill="#555">{better}</text>')
        c.save(path)


def sim2d_comparison(path, rows):
    names = [r['policy'] for r in rows]
    vals = [float(r['explore_time_s']) for r in rows]
    colors = {'ppo': PPO_C, 'heuristic': HEUR_C}
    c = Chart('2D simulator: time to fully explore the arena', 'time (s)', max(vals) * 1.15)
    gw = (W - L - R) / len(names)
    labels = {'ppo': 'PPO', 'heuristic': 'Heuristic', 'nearest': 'Nearest frontier', 'random': 'Random'}
    for k, (n, v) in enumerate(zip(names, vals)):
        x = L + k * gw + gw * 0.2
        c.add(f'<rect x="{x:.1f}" y="{c.y(v):.1f}" width="{gw*0.6:.1f}" height="{c.y(0)-c.y(v):.1f}" '
              f'fill="{colors.get(n, OTHER_C)}" fill-opacity="0.85"/>'
              f'<text x="{x+gw*0.3:.1f}" y="{c.y(v)-5:.1f}" text-anchor="middle">{v:.1f} s</text>'
              f'<text x="{x+gw*0.3:.1f}" y="{H-B+18}" text-anchor="middle">{labels.get(n, n)}</text>')
    c.add(f'<text x="{W-R}" y="{T-6}" text-anchor="end" font-size="11" fill="#555">lower is better</text>')
    c.save(path)


def paired_differences(path, ppo, heur):
    d = [m_p['area_240s'] - m_h['area_240s'] for m_p, m_h in zip(ppo, heur)]
    c = Chart('Gazebo: PPO minus heuristic, known area at 240 s (per paired run)',
              'difference (m²)', max(d) * 1.2, min(0, min(d) * 1.2), 'paired run')
    gw = (W - L - R) / len(d)
    for k, v in enumerate(d):
        x = L + k * gw + gw * 0.2
        y0, y1 = c.y(0), c.y(v)
        c.add(f'<rect x="{x:.1f}" y="{min(y0, y1):.1f}" width="{gw*0.6:.1f}" height="{abs(y0-y1):.1f}" '
              f'fill="{PPO_C if v > 0 else HEUR_C}" fill-opacity="0.85"/>'
              f'<text x="{x+gw*0.3:.1f}" y="{min(y0, y1)-5:.1f}" text-anchor="middle">{v:+.1f}</text>'
              f'<text x="{x+gw*0.3:.1f}" y="{H-B+18}" text-anchor="middle">{k+1}</text>')
    m = float(np.mean(d))
    c.add(f'<line x1="{L}" x2="{W-R}" y1="{c.y(m):.1f}" y2="{c.y(m):.1f}" stroke="#222" stroke-dasharray="5 4"/>'
          f'<text x="{L+6}" y="{c.y(m)-6:.1f}" text-anchor="start">mean {m:+.1f} m²</text>')
    c.add(f'<text x="{W-R}" y="{T-6}" text-anchor="end" font-size="11" fill="#555">'
          f'positive = PPO mapped more ({sum(v > 0 for v in d)}/{len(d)} pairs)</text>')
    c.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=os.path.join(REPO, 'models', 'ppo_frontier_policy.zip'))
    args = p.parse_args()
    out = os.path.join(REPO, 'graphs')
    res = os.path.join(REPO, 'results')
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(res, 'sim2d'), exist_ok=True)

    # 1. training curves
    hist = list(csv.DictReader(open(os.path.join(res, 'training', 'eval_history.csv'))))
    training_curve(os.path.join(out, '01_training_return.svg'),
                   os.path.join(out, '02_training_explore_time.svg'), hist)
    scalars = os.path.join(res, 'training', 'tensorboard_scalars.csv')
    if os.path.exists(scalars):
        training_metric_graphs(out, scalars, float(hist[0]['heuristic_return']))

    # 2. 2D simulator comparison (deterministic, recomputed from the model)
    from sb3_contrib import MaskablePPO
    from rl_sim.envs.frontier_env import FrontierExplorationEnv
    from rl_sim.train import run_episode
    env = FrontierExplorationEnv()
    model = MaskablePPO.load(args.model)
    rng = np.random.default_rng(0)
    pols = {'ppo': lambda o: int(model.predict(o, action_masks=env.action_masks(), deterministic=True)[0]),
            'heuristic': lambda _o: env.heuristic_action(),
            'nearest': lambda _o: env.nearest_action(),
            'random': lambda _o: int(rng.choice(np.flatnonzero(env.action_masks())))}
    rows = []
    for name, pol in pols.items():
        ret, t, n, _ = run_episode(env, pol)
        rows.append({'policy': name, 'return': round(ret, 4), 'explore_time_s': t, 'decisions': n})
    with open(os.path.join(res, 'sim2d', 'policy_comparison.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sim2d_comparison(os.path.join(out, '03_sim2d_time_to_explore.svg'), rows)

    # 3. Gazebo comparisons
    gz = os.path.join(res, 'gazebo_runs')
    import glob
    ppo = [load(f) for f in sorted(glob.glob(os.path.join(gz, 'ppo_*.csv')))]
    heur = [load(f) for f in sorted(glob.glob(os.path.join(gz, 'heuristic_*.csv')))]
    n = min(len(ppo), len(heur))
    ppo, heur = ppo[:n], heur[:n]
    mp, mh = [metrics(r) for r in ppo], [metrics(r) for r in heur]
    series = [('PPO', PPO_C), ('Heuristic', HEUR_C)]
    note = f'mean ± std over {n} runs per policy; dots = individual runs'

    coverage_svg(ppo, heur, os.path.join(out, '04_gazebo_coverage_over_time.svg'))

    keys = [f'area_{c}s' for c in CHECKPOINTS]
    bars(os.path.join(out, '05_gazebo_area_at_checkpoints.svg'),
         'Gazebo: known map area at each checkpoint', 'known area (m²)',
         [f'{c} s' for c in CHECKPOINTS], series,
         [[np.mean([m[k] for m in ms]) for k in keys] for ms in (mp, mh)],
         [[np.std([m[k] for m in ms], ddof=1) for k in keys] for ms in (mp, mh)],
         [[[m[k] for m in ms] for k in keys] for ms in (mp, mh)], note=note)

    paired_differences(os.path.join(out, '06_gazebo_paired_difference_240s.svg'), mp, mh)

    for fname, key, title, ylabel in (
            ('07_gazebo_time_to_30m2.svg', 'time_to_30m2', 'Gazebo: time to map 30 m² (lower is better)', 'time (s)'),
            ('08_gazebo_mean_area.svg', 'mean_area_0_240', 'Gazebo: mean known area over the 240 s run', 'mean area (m²)'),
            ('09_gazebo_failed_goals.svg', 'failed_goals', 'Gazebo: failed navigation goals per run', 'failed goals'),
            ('10_gazebo_decisions.svg', 'decisions', 'Gazebo: frontier decisions per run', 'decisions')):
        bars(os.path.join(out, fname), title, ylabel, [''], series,
             [[np.mean([m[key] for m in ms])] for ms in (mp, mh)],
             [[np.std([m[key] for m in ms], ddof=1)] for ms in (mp, mh)],
             [[[m[key] for m in ms]] for ms in (mp, mh)], note=note)
    print(f'Wrote {len(os.listdir(out))} graphs to {out}')


if __name__ == '__main__':
    main()
