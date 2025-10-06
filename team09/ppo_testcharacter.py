"""Minimal Stable Baselines3-powered TestCharacter for Bomberman variant 5."""

# This is necessary to find the main code
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / 'Bomberman'))
sys.path.insert(1, str(_THIS_DIR))
from entity import CharacterEntity  # noqa: E402

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - provide a clear error message
    raise SystemExit(
        "Stable Baselines3 is required. Install with `pip install stable-baselines3`."
    ) from exc

from project2.sb3_variant5_env import BombermanVariant5Env  # noqa: E402


class TestCharacter(CharacterEntity):
    """Directly feeds world observations into a trained PPO policy."""

    _ACTION_LOOKUP: Tuple[Tuple[int, int, bool], ...] = BombermanVariant5Env._ACTION_LOOKUP
    _OBS_CHANNELS = BombermanVariant5Env._OBS_CHANNELS
    _DEFAULT_LOCATIONS = (
        Path(__file__).with_name("models").joinpath("variant5_ppo_new.zip"),
        Path(__file__).parent / "project2" / "models" / "variant5_ppo_new.zip",
    )

    def __init__(
        self,
        name: str,
        avatar: str,
        x: int,
        y: int,
        *,
        model_path: Optional[Path] = None,
        deterministic: Optional[bool] = None,
    ):
        super().__init__(name, avatar, x, y)
        candidate = model_path or os.environ.get("SB3_MODEL_PATH")
        if candidate is None:
            for default in self._DEFAULT_LOCATIONS:
                if default.exists():
                    candidate = default
                    break
        if candidate is None:
            candidate = self._DEFAULT_LOCATIONS[0]
        resolved_path = Path(candidate).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Stable Baselines3 model not found at {resolved_path}. "
                "Set SB3_MODEL_PATH to override."
            )
        self._model = PPO.load(str(resolved_path), device="cpu")
        if deterministic is None:
            env_flag = os.environ.get("SB3_DETERMINISTIC", "1").lower()
            deterministic = env_flag not in {"0", "false", "no"}
        self._deterministic = deterministic
        self._max_time: Optional[int] = None

    def do(self, wrld) -> None:  # noqa: D401
        if wrld is None:
            return
        if self._max_time is None and wrld is not None:
            self._max_time = max(1, wrld.time)
        obs = self._encode_observation(wrld)
        action, _ = self._model.predict(obs, deterministic=self._deterministic)
        dx, dy, place_bomb = self._ACTION_LOOKUP[int(action)]
        self.move(dx, dy)
        if place_bomb:
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
        max_time = float(self._max_time or wrld.time or 1)
        time_norm = np.array([wrld.time / max_time], dtype=np.float32)
        score_norm = np.array([wrld.scores.get(self.name, 0) / 200.0], dtype=np.float32)
        flat = [grids[channel].ravel() for channel in self._OBS_CHANNELS]
        return np.concatenate(flat + [time_norm, score_norm]).astype(np.float32)


__all__ = ["TestCharacter"]
