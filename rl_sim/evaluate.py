"""
Compare a trained policy with the heuristic, nearest-frontier and random
choice on the same fixed world, and save the explored map of each.

  cd ~/swarm && python3 rl_sim/evaluate.py --model ~/rl_sim_runs/<run>/best_model.zip --world-seed 42
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rl_sim.envs.frontier_env import FrontierExplorationEnv  # noqa: E402
from rl_sim.train import run_episode  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', help='MaskablePPO .zip (omit to show only the baselines)')
    p.add_argument('--world-seed', type=int, default=42)
    p.add_argument('--out-dir', default=None, help='where to save final-map PNGs')
    args = p.parse_args()

    env = FrontierExplorationEnv(args.world_seed)
    rng = np.random.default_rng(0)
    policies = {
        'heuristic': lambda _o: env.heuristic_action(),
        'nearest': lambda _o: env.nearest_action(),
        'random': lambda _o: int(rng.choice(np.flatnonzero(env.action_masks()))),
    }
    if args.model:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(os.path.expanduser(args.model))
        policies = {'ppo': lambda o: int(model.predict(o, action_masks=env.action_masks(),
                                                       deterministic=True)[0]), **policies}
    out_dir = args.out_dir or (os.path.dirname(os.path.expanduser(args.model)) if args.model else '.')
    print(f'World {args.world_seed}:')
    for name, pol in policies.items():
        ret, t, n, end = run_episode(env, pol)
        print(f'  {name:9s} return {ret:6.3f} | explored in {t:5.1f} s | {n} decisions | {end}')
        img = cv2.resize(env.render(), (510, 510), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, f'final_map_{name}.png'), img)
    print(f'Final maps saved in {out_dir}')


if __name__ == '__main__':
    main()
