"""
Analyse paired Gazebo runs from rl_sim/gazebo_test.sh (PPO vs heuristic).

  python3 rl_sim/analyze_gazebo_tests.py

Runs are paired in chronological order (k-th PPO run with k-th heuristic
run; gazebo_test.sh runs are alternated, so pairs share machine state).
Statistics: mean ± std, paired differences, and an exact two-sided paired
permutation (sign-flip) test, which needs no normality assumption or SciPy.
"""
import argparse
import csv
import glob
import itertools
import os

import numpy as np

CHECKPOINTS = (60, 120, 180, 240)
AREA_TARGET = 30.0      # m², for the time-to-area metric


def load(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r['sim_time_s']) for r in rows])
    a = np.array([float(r['known_m2']) for r in rows])
    last = rows[-1]
    return {'t': t, 'a': a, 'decisions': int(last['decisions']),
            'failed': int(last['failed_goals']), 'name': os.path.basename(path)}


def area_at(run, sec):
    i = np.searchsorted(run['t'], sec)
    return float(run['a'][min(i, len(run['a']) - 1)])


def metrics(run, horizon=240):
    t, a = run['t'], run['a']
    m = {f'area_{c}s': area_at(run, c) for c in CHECKPOINTS}
    keep = t <= horizon
    m['mean_area_0_240'] = float(getattr(np, 'trapezoid', getattr(np, 'trapz', None))(a[keep], t[keep]) / horizon)
    reach = np.nonzero(a >= AREA_TARGET)[0]
    m['time_to_30m2'] = float(t[reach[0]]) if len(reach) else float('nan')
    m['failed_goals'] = run['failed']
    m['decisions'] = run['decisions']
    return m


def perm_test(diffs):
    """Exact two-sided paired permutation p-value for mean(diffs) != 0."""
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    obs = abs(d.mean())
    count = total = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        total += 1
        if abs((d * signs).mean()) >= obs - 1e-12:
            count += 1
    return count / total


def svg(ppo, heur, path, horizon=240):
    W, H, L, B, T, R = 640, 420, 60, 50, 30, 20
    grid = np.arange(0, horizon + 1, 2.0)
    ymax = max(max(r['a'].max() for r in ppo + heur), 1) * 1.1

    def xs(t):
        return L + (W - L - R) * t / horizon

    def ys(v):
        return H - B - (H - B - T) * v / ymax

    def curves(runs):
        return np.array([np.interp(grid, r['t'], r['a']) for r in runs])

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W/2}" y="16" text-anchor="middle" font-size="15" font-weight="bold">Gazebo: known map area over time</text>']
    for v in range(0, int(ymax) + 1, 10):
        out.append(f'<line x1="{L}" x2="{W-R}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="#ddd"/>'
                   f'<text x="{L-8}" y="{ys(v)+4:.1f}" text-anchor="end">{v}</text>')
    for t in range(0, horizon + 1, 60):
        out.append(f'<text x="{xs(t):.1f}" y="{H-B+18}" text-anchor="middle">{t}</text>')
    out.append(f'<text x="{(L+W-R)/2}" y="{H-12}" text-anchor="middle">simulated time (s)</text>'
               f'<text x="16" y="{(H-B)/2}" transform="rotate(-90 16 {(H-B)/2})" text-anchor="middle">known area (m²)</text>')
    for runs, color, label, ly in ((ppo, '#1f77b4', 'PPO policy', 46), (heur, '#d62728', 'Heuristic (frontier_coordinator cost)', 64)):
        c = curves(runs)
        lo, hi, mean = c.min(0), c.max(0), c.mean(0)
        band = ' '.join(f'{xs(t):.1f},{ys(v):.1f}' for t, v in zip(grid, hi)) + ' ' + \
               ' '.join(f'{xs(t):.1f},{ys(v):.1f}' for t, v in zip(grid[::-1], lo[::-1]))
        out.append(f'<polygon points="{band}" fill="{color}" fill-opacity="0.15"/>')
        line = ' '.join(f'{xs(t):.1f},{ys(v):.1f}' for t, v in zip(grid, mean))
        out.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        out.append(f'<line x1="{L+15}" x2="{L+40}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>'
                   f'<text x="{L+46}" y="{ly+4}">{label}: mean of {len(runs)} runs, band = min–max</text>')
    out.append('</svg>')
    with open(path, 'w') as fh:
        fh.write('\n'.join(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'gazebo_runs'))
    p.add_argument('--svg', default=None)
    args = p.parse_args()
    ppo = [load(f) for f in sorted(glob.glob(f'{args.dir}/ppo_*.csv'))]
    heur = [load(f) for f in sorted(glob.glob(f'{args.dir}/heuristic_*.csv'))]
    n = min(len(ppo), len(heur))
    ppo, heur = ppo[:n], heur[:n]
    print(f'{n} paired runs\n')
    mp = [metrics(r) for r in ppo]
    mh = [metrics(r) for r in heur]
    keys = list(mp[0].keys())
    print('| Metric | PPO mean ± std | Heuristic mean ± std | Mean paired diff (PPO−H) | PPO better in | p (exact paired permutation) |')
    print('|---|---|---|---|---|---|')
    for k in keys:
        a = np.array([m[k] for m in mp], float)
        b = np.array([m[k] for m in mh], float)
        d = a - b
        lower_better = k in ('time_to_30m2', 'failed_goals')
        wins = int(np.sum(d < 0) if lower_better else np.sum(d > 0))
        print(f'| {k} | {np.nanmean(a):.2f} ± {np.nanstd(a, ddof=1):.2f} | {np.nanmean(b):.2f} ± {np.nanstd(b, ddof=1):.2f} '
              f'| {np.nanmean(d):+.2f} | {wins}/{n} | {perm_test(d):.4f} |')
    print('\nPer-run (area at 60/120/180/240 s, failed goals):')
    for i, (a, b) in enumerate(zip(ppo, heur), 1):
        fa = ' / '.join(f'{area_at(a, c):.1f}' for c in CHECKPOINTS)
        fb = ' / '.join(f'{area_at(b, c):.1f}' for c in CHECKPOINTS)
        print(f'  pair {i}: PPO {fa} ({a["failed"]} failed) | heuristic {fb} ({b["failed"]} failed)')
    if args.svg:
        svg(ppo, heur, args.svg)
        print(f'\nCoverage plot: {args.svg}')


if __name__ == '__main__':
    main()
