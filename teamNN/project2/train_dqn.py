import os
import argparse

import torch
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
import inspect
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from .env import BombermanProject2Env


def make_env(map_path: str):
    def _thunk():
        # Use environment defaults (bottom_start=True, randomize=True, simple_reward=True)
        env = BombermanProject2Env(map_path=map_path, render_mode=None)
        env = Monitor(env)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default=os.path.join(os.path.dirname(__file__), 'map.txt'))
    parser.add_argument('--timesteps', type=int, default=600000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_path', type=str, default=os.path.join(os.path.dirname(__file__), 'dqn_model'))
    args = parser.parse_args()

    env = DummyVecEnv([make_env(args.map)])

    # Enable dueling when supported by installed SB3 version
    supports_dueling = 'dueling' in inspect.signature(DQNPolicy.__init__).parameters
    policy_kwargs = dict(net_arch=[256, 256])
    if supports_dueling:
        policy_kwargs['dueling'] = True
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=1e-4,
        buffer_size=500000,
        learning_starts=30000,
        batch_size=512,
        tau=1.0,
        gamma=0.9999,
        train_freq=4,
        target_update_interval=2000,
        exploration_fraction=0.7,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(os.path.dirname(__file__), 'tb'),
        device='auto'
    )

    eval_env = DummyVecEnv([make_env(args.map)])
    eval_callback = EvalCallback(eval_env, best_model_save_path=args.save_path,
                                 log_path=args.save_path, eval_freq=10000,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    os.makedirs(args.save_path, exist_ok=True)
    model.save(os.path.join(args.save_path, 'final'))


if __name__ == '__main__':
    main()


