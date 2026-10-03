"""
Train a frontier-selection policy with MaskablePPO in the 2D simulator.

  cd ~/swarm/final-product && python3 rl_sim/train.py

The best policy (by deterministic evaluation) is saved to
<run-dir>/best_model.zip; progress is printed, written to
<run-dir>/eval_history.csv and logged to TensorBoard (<run-dir>/tb).
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor  # noqa: E402

from rl_sim.core.world import DEFAULT_WORLD  # noqa: E402
from rl_sim.envs.frontier_env import FrontierExplorationEnv  # noqa: E402


def run_episode(env, policy):
    """Return (return, sim_time_s, decisions, end_reason) of one episode."""
    obs, _ = env.reset()
    ret, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        ret += r
        done = term or trunc
    return ret, info['sim_time_s'], info['decisions'], info['end_reason']


class EvalCallback(BaseCallback):
    """Every eval_every decisions: run the deterministic policy, keep the best model."""

    def __init__(self, world, eval_every, run_dir, baseline):
        super().__init__()
        self.env = FrontierExplorationEnv(world)
        self.csv_path = os.path.join(run_dir, 'eval_history.csv')
        with open(self.csv_path, 'w', newline='') as fh:
            csv.writer(fh).writerow(['decisions', 'return', 'explore_time_s', 'heuristic_return', 'heuristic_explore_time_s'])
        self.eval_every = eval_every
        self.run_dir = run_dir
        self.baseline = baseline
        self.best = -np.inf
        self.history = []
        self.next_eval = 0

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval = self.num_timesteps + self.eval_every
        env = self.env

        def policy(obs):
            return int(self.model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0])

        ret, t, n, end = run_episode(env, policy)
        self.history.append(ret)
        self.logger.record('eval/return', ret)
        self.logger.record('eval/explore_time_s', t)
        mark = ''
        if ret > self.best:
            self.best = ret
            self.model.save(os.path.join(self.run_dir, 'best_model'))
            mark = '  <- best, saved'
        h_ret, h_t = self.baseline
        with open(self.csv_path, 'a', newline='') as fh:
            csv.writer(fh).writerow([self.num_timesteps, round(ret, 4), round(t, 1), round(h_ret, 4), round(h_t, 1)])
        print(f'[{self.num_timesteps:>7} decisions] policy: return {ret:6.3f}, explored in {t:5.1f} s '
              f'({n} decisions, {end}) | heuristic {h_ret:6.3f} / {h_t:.1f} s{mark}', flush=True)
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--world', type=int, default=DEFAULT_WORLD, help='arena number (default: benchmark arena)')
    p.add_argument('--timesteps', type=int, default=300_000, help='total decisions')
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=10_000)
    p.add_argument('--seed', type=int, default=0, help='PPO seed')
    p.add_argument('--run-dir', default=None)
    args = p.parse_args()
    run_dir = args.run_dir or os.path.expanduser(
        f'~/rl_sim_runs/ppo_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir, exist_ok=True)

    probe = FrontierExplorationEnv(args.world)
    h_ret, h_t, _, _ = run_episode(probe, lambda _o: probe.heuristic_action())
    print(f'Heuristic (frontier_coordinator) baseline: return {h_ret:.3f}, '
          f'explores in {h_t:.1f} s', flush=True)

    fns = [lambda: FrontierExplorationEnv(args.world) for _ in range(args.num_envs)]
    venv = SubprocVecEnv(fns) if args.num_envs > 1 else DummyVecEnv(fns)
    venv = VecMonitor(venv)
    venv.seed(args.seed)

    # Settings for a small, deterministic, short-horizon task (episodes are
    # ~10-20 decisions): large batches for low-variance gradients, entropy
    # bonus to keep exploring slot choices early, linearly decaying lr so the
    # policy settles at the end.
    model = MaskablePPO(
        'MlpPolicy', venv,
        n_steps=256, batch_size=256, n_epochs=10,
        learning_rate=lambda progress: 3e-4 * progress,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        tensorboard_log=os.path.join(run_dir, 'tb'), seed=args.seed, verbose=0)

    cb = EvalCallback(args.world, args.eval_every, run_dir, (h_ret, h_t))
    print(f'Run dir: {run_dir}', flush=True)
    try:
        model.learn(total_timesteps=args.timesteps, callback=cb)
    except KeyboardInterrupt:
        print('Interrupted.', flush=True)
    finally:
        model.save(os.path.join(run_dir, 'final_model'))
        venv.close()
        try:   # all TensorBoard scalars as CSV, for rl_sim/make_graphs.py
            from rl_sim.export_tensorboard import export
            export(os.path.join(run_dir, 'tb'), os.path.join(run_dir, 'tensorboard_scalars.csv'))
        except Exception as e:  # graphs are optional; never lose a trained model over them
            print(f'(TensorBoard export skipped: {e})', flush=True)
    print(f'Best deterministic return {cb.best:.3f} (heuristic {h_ret:.3f}). '
          f'Models: {run_dir}/best_model.zip, final_model.zip', flush=True)


if __name__ == '__main__':
    main()
