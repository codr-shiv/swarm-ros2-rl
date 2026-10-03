"""
Count Nav2 "robot stuck" messages for each Gazebo run.

  python3 rl_sim/count_nav2_failures.py results/gazebo_runs [out.csv]

For every <policy>_<time>.csv in the folder, reads the matching launch log
~/rl_sim_runs/gazebo_tests/<policy>_<time>/nav2_bringup_multi.launch.py.log
(written by gazebo_test.sh) and counts the messages that indicate a robot
pinned against an obstacle.
"""
import csv
import glob
import os
import sys

MESSAGES = {
    'start_or_goal_in_obstacle': 'Either of the start or goal pose are an obstacle',
    'no_valid_trajectories': 'No valid trajectories',
    'backup_collision_ahead': 'Collision Ahead',
}
LOG_ROOT = os.path.expanduser('~/rl_sim_runs/gazebo_tests')


def count(run_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, '*.csv'))):
        tag = os.path.basename(path)[:-4]
        log = os.path.join(LOG_ROOT, tag, 'nav2_bringup_multi.launch.py.log')
        if not os.path.exists(log):
            continue
        text = open(log, errors='ignore').read()
        rows.append({'run': tag, 'policy': tag.split('_')[0],
                     **{k: text.count(m) for k, m in MESSAGES.items()}})
    return rows


if __name__ == '__main__':
    rows = count(sys.argv[1] if len(sys.argv) > 1 else 'results/gazebo_runs')
    for pol in ('ppo', 'heuristic'):
        sel = [r for r in rows if r['policy'] == pol]
        if sel:
            print(f"{pol:9s} runs={len(sel)} " + ' '.join(
                f"{k}={sum(r[k] for r in sel) / len(sel):.1f}/run" for k in MESSAGES))
    if len(sys.argv) > 2 and rows:
        with open(sys.argv[2], 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print('Wrote', sys.argv[2])
