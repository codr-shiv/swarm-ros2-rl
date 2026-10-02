"""
Train a frontier-selection policy with MaskablePPO on one fixed world.

  cd ~/swarm && python3 rl_sim/train.py --world-seed 42

The world is the Gazebo world for GAZEBO_WORLD_SEED=<world-seed>. The best
policy (by deterministic evaluation) is saved to <run-dir>/best_model.zip;
progress is printed and logged to TensorBoard (<run-dir>/tb).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor  # noqa: E402

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

    def __init__(self, world_seed, eval_every, run_dir, baseline):
        super().__init__()
        self.env = FrontierExplorationEnv(world_seed)
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
        print(f'[{self.num_timesteps:>7} decisions] policy: return {ret:6.3f}, explored in {t:5.1f} s '
              f'({n} decisions, {end}) | heuristic {h_ret:6.3f} / {h_t:.1f} s{mark}', flush=True)
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--world-seed', type=int, default=42, help='GAZEBO_WORLD_SEED of the world')
    p.add_argument('--timesteps', type=int, default=300_000, help='total decisions')
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=10_000)
    p.add_argument('--seed', type=int, default=0, help='PPO seed')
    p.add_argument('--run-dir', default=None)
    args = p.parse_args()
    run_dir = args.run_dir or os.path.expanduser(
        f'~/rl_sim_runs/world{args.world_seed}_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir, exist_ok=True)

    probe = FrontierExplorationEnv(args.world_seed)
    h_ret, h_t, _, _ = run_episode(probe, lambda _o: probe.heuristic_action())
    print(f'World {args.world_seed}: heuristic (frontier_coordinator) return {h_ret:.3f}, '
          f'explores in {h_t:.1f} s', flush=True)

    fns = [lambda: FrontierExplorationEnv(args.world_seed) for _ in range(args.num_envs)]
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

    cb = EvalCallback(args.world_seed, args.eval_every, run_dir, (h_ret, h_t))
    print(f'Run dir: {run_dir}', flush=True)
    try:
        model.learn(total_timesteps=args.timesteps, callback=cb)
    except KeyboardInterrupt:
        print('Interrupted.', flush=True)
    finally:
        model.save(os.path.join(run_dir, 'final_model'))
        venv.close()
    print(f'Best deterministic return {cb.best:.3f} (heuristic {h_ret:.3f}). '
          f'Models: {run_dir}/best_model.zip, final_model.zip', flush=True)


if __name__ == '__main__':
    main()
