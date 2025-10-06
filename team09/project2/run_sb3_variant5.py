"""Stable Baselines3 helpers for Bomberman variant 5."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from sb3_variant5_env import BombermanVariant5Env, _USES_GYMNASIUM_API

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - user feedback path
    raise SystemExit(
        "Stable Baselines3 is required. Install with `pip install stable-baselines3`."
    ) from exc


def train(model_path: Path, timesteps: int, resume_from: Optional[Path] = None) -> None:
    env = BombermanVariant5Env()
    if resume_from and resume_from.exists():
        model = PPO.load(str(resume_from), env=env, print_system_info=True)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            gamma=0.99,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
        )
    model.learn(total_timesteps=timesteps)
    model.save(str(model_path))
    env.close()


def evaluate(model_path: Path, episodes: int, render: bool = False) -> None:
    env = BombermanVariant5Env(render_mode="console" if render else None)
    model = PPO.load(str(model_path), env=env, print_system_info=True)
    rewards = []
    successes = 0
    for _ in range(episodes):
        if _USES_GYMNASIUM_API:
            obs, _ = env.reset()
            terminated = False
            truncated = False
        else:
            obs = env.reset()
            terminated = False
            truncated = False
        ep_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            if _USES_GYMNASIUM_API:
                obs, reward, terminated, truncated, info = step_result
                success = bool(info.get("success"))
            else:
                obs, reward, done, info = step_result
                terminated = bool(done)
                truncated = False
                success = bool(info.get("success"))
            ep_reward += float(reward)
            if success and terminated:
                successes += 1
        rewards.append(ep_reward)
    env.close()
    print(f"Episodes: {episodes}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Successes: {successes} / {episodes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a PPO policy")
    train_parser.add_argument("model", type=Path, help="Path to save the trained model")
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000,
        help="Number of training timesteps (default: 10000)",
    )
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint to resume training from",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained policy")
    eval_parser.add_argument("model", type=Path, help="Path to the trained model")
    eval_parser.add_argument(
        "--episodes", type=int, default=5, help="Number of evaluation episodes"
    )
    eval_parser.add_argument(
        "--render", action="store_true", help="Render the board to the console"
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args.model, args.timesteps, args.resume_from)
    elif args.command == "eval":
        evaluate(args.model, args.episodes, render=args.render)
    else:  # pragma: no cover - defensive programming
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
