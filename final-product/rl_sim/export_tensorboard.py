"""
Export every scalar from a TensorBoard log to a CSV (tag, step, value).

  python3 rl_sim/export_tensorboard.py <run-dir or tb dir> [results/training/tensorboard_scalars.csv]

train.py calls this automatically at the end of training.
"""
import csv
import glob
import os
import sys


def export(log_dir, out_csv):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = sorted(glob.glob(os.path.join(log_dir, '**', 'events.out.tfevents.*'), recursive=True))
    if not files:
        raise FileNotFoundError(f'no TensorBoard event file under {log_dir}')
    ea = EventAccumulator(files[-1], size_guidance={'scalars': 0})
    ea.Reload()
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['tag', 'step', 'value'])
        for tag in ea.Tags()['scalars']:
            for e in ea.Scalars(tag):
                w.writerow([tag, e.step, e.value])
    return out_csv


if __name__ == '__main__':
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, 'results', 'training', 'tensorboard')
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(repo, 'results', 'training', 'tensorboard_scalars.csv')
    print('Wrote', export(os.path.expanduser(src), dst))
