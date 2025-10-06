"""Stable-Baselines3 driven character for Bomberman variant 5."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from entity import CharacterEntity
from sb3_variant5_env import BombermanVariant5Env

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - explicit feedback when missing dep
    raise SystemExit(
        "Stable Baselines3 is required. Install with `pip install stable-baselines3`."
    ) from exc


class Variant5PPOCharacter(CharacterEntity):
    """Character that mirrors the Stable Baselines3 observation/action spaces."""

    _ACTION_LOOKUP: Tuple[Tuple[int, int, bool], ...] = BombermanVariant5Env._ACTION_LOOKUP
    _OBS_CHANNELS = BombermanVariant5Env._OBS_CHANNELS

    def __init__(
        self,
        name: str,
        avatar: str,
        x: int,
        y: int,
        model_path: Path,
        *,
        deterministic: bool,
        max_time: int,
    ):
        super().__init__(name, avatar, x, y)
        self._model = PPO.load(str(model_path), device="cpu")
        self._deterministic = deterministic
        self._max_time = max(1, max_time)

    def do(self, wrld) -> None:  # noqa: D401 (custom behaviour)
        if wrld is None:
            return
        obs = self._encode_observation(wrld)
        action, _ = self._model.predict(obs, deterministic=self._deterministic)
        dx, dy, drop_bomb = self._ACTION_LOOKUP[int(action)]
        self.move(dx, dy)
        if drop_bomb:
            self.place_bomb()

    def _encode_observation(self, wrld) -> np.ndarray:
        width = wrld.width()
        height = wrld.height()
        grids = {
            channel: np.zeros((height, width), dtype=np.float32)
            for channel in self._OBS_CHANNELS
        }
        for x in range(width):
            for y in range(height):
                if wrld.wall_at(x, y):
                    grids["walls"][y, x] = 1.0
                if wrld.exit_at(x, y):
                    grids["exit"][y, x] = 1.0
                if wrld.bomb_at(x, y):
                    grids["bombs"][y, x] = 1.0
                if wrld.explosion_at(x, y):
                    grids["explosions"][y, x] = 1.0
                if wrld.monsters_at(x, y):
                    grids["monsters"][y, x] = 1.0
                chars = wrld.characters_at(x, y)
                if chars:
                    for ch in chars:
                        if ch.name == self.name:
                            grids["agent"][y, x] = 1.0
        time_norm = np.array([wrld.time / float(self._max_time)], dtype=np.float32)
        score_norm = np.array([wrld.scores.get(self.name, 0) / 200.0], dtype=np.float32)
        flat_channels = [grids[ch].ravel() for ch in self._OBS_CHANNELS]
        return np.concatenate(flat_channels + [time_norm, score_norm]).astype(np.float32)


__all__ = ["Variant5PPOCharacter"]
