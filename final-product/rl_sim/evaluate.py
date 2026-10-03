"""
Compare a trained policy with the heuristic, nearest-frontier and random
choice in the 2D simulator, and save the explored map of each.

  cd ~/swarm/final-product && python3 rl_sim/evaluate.py --model models/ppo_frontier_policy.zip
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rl_sim.core.world import DEFAULT_WORLD  # noqa: E402
from rl_sim.envs.frontier_env import FrontierExplorationEnv  # noqa: E402
from rl_sim.train import run_episode  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', help='MaskablePPO .zip (omit to show only the baselines)')
    p.add_argument('--world', type=int, default=DEFAULT_WORLD, help='arena number')
    p.add_argument('--out-dir', default=None, help='where to save final-map PNGs (and videos)')
    p.add_argument('--video', action='store_true', help='also record an MP4 of each episode')
    args = p.parse_args()

    env = FrontierExplorationEnv(args.world)
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
    out_dir = os.path.expanduser(args.out_dir or '~/rl_sim_runs/final_maps')
    os.makedirs(out_dir, exist_ok=True)
    print('2D simulator results:')
    for name, pol in policies.items():
        writer = None
        if args.video:
            writer = cv2.VideoWriter(os.path.join(out_dir, f'sim2d_{name}.mp4'),
                                     cv2.VideoWriter_fourcc(*'mp4v'), 10, (510, 510))

            def frame(e, writer=writer, name=name):
                img = cv2.resize(e.render(), (510, 510), interpolation=cv2.INTER_NEAREST)
                cv2.putText(img, f'{name}  t={e.t:5.1f}s  known={e._known_m2():4.1f} m2', (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                writer.write(img)
            env.on_tick = frame
        ret, t, n, end = run_episode(env, pol)
        env.on_tick = None
        if writer is not None:
            writer.release()
        print(f'  {name:9s} return {ret:6.3f} | explored in {t:5.1f} s | {n} decisions | {end}')
        img = cv2.resize(env.render(), (510, 510), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, f'final_map_{name}.png'), img)
    print(f'Final maps saved in {out_dir}')


if __name__ == '__main__':
    main()
