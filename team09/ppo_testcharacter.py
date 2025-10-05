"""Stable Baselines3-powered TestCharacter for Bomberman variant 5."""

# This is necessary to find the main code
import sys
import os
from collections import deque
from pathlib import Path
from typing import Iterable, Optional, Tuple

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
    """Drop-in replacement that mirrors the SB3 training interface."""

    _ACTION_LOOKUP: Tuple[Tuple[int, int, bool], ...] = BombermanVariant5Env._ACTION_LOOKUP
    _OBS_CHANNELS = BombermanVariant5Env._OBS_CHANNELS
    _DEFAULT_LOCATIONS = (
        Path(__file__).with_name("models").joinpath("variant5_ppo.zip"),
        Path(__file__).parent / "project2" / "models" / "variant5_ppo.zip",
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

        me = wrld.me(self)
        bombs = list(getattr(wrld, "bombs", {}).values())
        monster_threat = self._collect_monster_threats(wrld)
        bomb_threat = self._collect_danger_tiles(wrld, bombs)
        danger_tiles = bomb_threat | monster_threat
        owned_bombs = [b for b in bombs if getattr(b.owner, "name", None) == self.name]

        if place_bomb and not self._should_place_bomb(wrld, me, monster_threat):
            place_bomb = False

        escape_move = None
        if owned_bombs or (me.x, me.y) in danger_tiles:
            escape_move = self._find_escape_move(wrld, me, danger_tiles, bombs, monster_threat)

        exit_move = None
        if wrld.exitcell and escape_move is None and (me.x, me.y) not in danger_tiles:
            exit_move = self._next_step_toward_exit(wrld, me, danger_tiles)

        if escape_move is not None:
            dx, dy = escape_move
        elif exit_move is not None:
            ex, ey = wrld.exitcell
            current_dist = abs(me.x - ex) + abs(me.y - ey)
            rl_next_dist = abs((me.x + dx) - ex) + abs((me.y + dy) - ey)
            exit_dx, exit_dy = exit_move
            exit_next_dist = abs((me.x + exit_dx) - ex) + abs((me.y + exit_dy) - ey)
            if exit_next_dist < current_dist and exit_next_dist <= rl_next_dist:
                dx, dy = exit_dx, exit_dy

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

    def _should_place_bomb(self, wrld, me, monster_threat: set) -> bool:
        for dx, dy in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
            (2, 0),
            (-2, 0),
            (0, 2),
            (0, -2),
        ):
            nx = me.x + dx
            ny = me.y + dy
            if nx < 0 or nx >= wrld.width() or ny < 0 or ny >= wrld.height():
                continue
            if wrld.wall_at(nx, ny):
                return True
            if wrld.monsters_at(nx, ny) or (nx, ny) in monster_threat:
                return True
        return False

    def _collect_danger_tiles(self, wrld, bombs: Iterable) -> set:
        danger = set()
        for bomb in bombs:
            danger.update(self._blast_tiles(wrld, bomb))
        return danger

    def _blast_tiles(self, wrld, bomb) -> set:
        tiles = {(bomb.x, bomb.y)}
        width = wrld.width()
        height = wrld.height()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            xx, yy = bomb.x, bomb.y
            for _ in range(wrld.expl_range):
                xx += dx
                yy += dy
                if xx < 0 or xx >= width or yy < 0 or yy >= height:
                    break
                if wrld.exit_at(xx, yy):
                    break
                tiles.add((xx, yy))
                if wrld.wall_at(xx, yy) or wrld.bomb_at(xx, yy):
                    break
        return tiles

    def _find_escape_move(
        self,
        wrld,
        me,
        danger_tiles: set,
        bombs: Iterable,
        monster_threat: set,
    ) -> Optional[Tuple[int, int]]:
        width = wrld.width()
        height = wrld.height()
        visited = set()
        queue = deque()
        queue.append(((me.x, me.y), (0, 0)))
        visited.add((me.x, me.y))
        best_move = None
        best_score = -1.0

        while queue:
            (cx, cy), first_step = queue.popleft()
            current_first = first_step
            for dx, dy in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                nx = cx + dx
                ny = cy + dy
                step = first_step
                if (cx, cy) == (me.x, me.y):
                    step = (dx, dy)
                if (nx, ny) in visited:
                    continue
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if (
                    wrld.wall_at(nx, ny)
                    or wrld.bomb_at(nx, ny)
                    or wrld.explosion_at(nx, ny)
                    or (nx, ny) in danger_tiles
                ):
                    continue
                if wrld.monsters_at(nx, ny) or (nx, ny) in monster_threat:
                    continue
                visited.add((nx, ny))
                score = self._safety_score((nx, ny), bombs, monster_threat)
                if score > best_score:
                    best_score = score
                    best_move = step
                queue.append(((nx, ny), step))
        if best_move in (None, (0, 0)):
            return None
        dx, dy = best_move
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))
        return dx, dy

    def _safety_score(
        self,
        pos: Tuple[int, int],
        bombs: Iterable,
        monster_threat: set,
    ) -> float:
        px, py = pos
        bomb_dist = min(
            (abs(px - b.x) + abs(py - b.y) for b in bombs),
            default=10.0,
        )
        monster_dist = min(
            (abs(px - mx) + abs(py - my) for mx, my in monster_threat),
            default=10.0,
        )
        return bomb_dist * 2.0 + monster_dist

    def _next_step_toward_exit(self, wrld, me, danger_tiles: set) -> Optional[Tuple[int, int]]:
        ex, ey = wrld.exitcell
        width = wrld.width()
        height = wrld.height()
        visited = set()
        queue = deque()
        queue.append(((me.x, me.y), (0, 0)))
        visited.add((me.x, me.y))

        while queue:
            (cx, cy), first_step = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = cx + dx
                ny = cy + dy
                if (nx, ny) in visited:
                    continue
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if wrld.wall_at(nx, ny) or wrld.bomb_at(nx, ny) or wrld.explosion_at(nx, ny):
                    continue
                if (nx, ny) in danger_tiles and (nx, ny) != (ex, ey):
                    continue
                if wrld.monsters_at(nx, ny) and (nx, ny) != (ex, ey):
                    continue
                step = first_step if first_step != (0, 0) else (dx, dy)
                if (nx, ny) == (ex, ey):
                    return step
                visited.add((nx, ny))
                queue.append(((nx, ny), step))
        return None

    def _collect_monster_threats(self, wrld) -> set:
        threats = set()
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                monsters = wrld.monsters_at(x, y)
                if not monsters:
                    continue
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                            threats.add((nx, ny))
                for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                        threats.add((nx, ny))
        return threats


__all__ = ["TestCharacter"]
