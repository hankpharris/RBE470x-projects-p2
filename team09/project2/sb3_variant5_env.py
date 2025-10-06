"""Gym-compatible environment for Bomberman project 2, variant 5."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import math

import numpy as np

try:  # Prefer gymnasium when available (Stable Baselines3 >= 2.0)
    import gymnasium as gym
    from gymnasium import spaces
    _USES_GYMNASIUM_API = True
except ImportError:  # Fallback to legacy gym API
    import gym
    from gym import spaces
    _USES_GYMNASIUM_API = False

# Make sure the Bomberman package is importable
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_BOMBERMAN_DIR = _REPO_ROOT / "Bomberman"
if str(_BOMBERMAN_DIR) not in sys.path:
    sys.path.insert(0, str(_BOMBERMAN_DIR))

from entity import CharacterEntity  # pylint: disable=wrong-import-position
from events import Event  # pylint: disable=wrong-import-position
from real_world import RealWorld  # pylint: disable=wrong-import-position
from monsters.stupid_monster import StupidMonster  # pylint: disable=wrong-import-position
from monsters.selfpreserving_monster import SelfPreservingMonster  # pylint: disable=wrong-import-position


class RLCharacter(CharacterEntity):
    """Character controlled directly by the reinforcement learning algorithm."""

    def __init__(self, name: str, avatar: str, x: int, y: int):
        super().__init__(name, avatar, x, y)
        self.is_done: bool = False
        self.exit_reached: bool = False

    def reset_flags(self) -> None:
        self.is_done = False
        self.exit_reached = False

    def do(self, wrld) -> None:  # noqa: D401 (inherited docstring)
        # The RL loop sets actions explicitly via env.step(); nothing to do here.
        return

    def done(self, wrld) -> None:  # noqa: D401 (inherited docstring)
        # Called by the engine when the character dies or exits.
        self.is_done = True
        # Wrld can be None when called from sensed-world contexts, so guard it.
        if wrld is not None:
            me = wrld.me(self)
            self.exit_reached = bool(me and wrld.exit_at(me.x, me.y))


class BombermanVariant5Env(gym.Env):
    """Gym/Gymnasium environment wrapper around Bomberman project 2 variant 5."""

    metadata = {"render_modes": ["console"], "render_fps": 4}

    # (dx, dy, place_bomb)
    _ACTION_LOOKUP: Sequence[Tuple[int, int, bool]] = (
        (0, 0, False),  # idle
        (1, 0, False),  # right
        (-1, 0, False),  # left
        (0, 1, False),  # down
        (0, -1, False),  # up
        (1, 0, True),  # right + bomb
        (-1, 0, True),  # left + bomb
        (0, 1, True),  # down + bomb
        (0, -1, True),  # up + bomb
        (0, 0, True),  # drop bomb while standing still
    )

    _OBS_CHANNELS = ("walls", "exit", "bombs", "explosions", "monsters", "agent")

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self._map_path = _THIS_FILE.with_name("map.txt")
        self._map_info = self._parse_map(self._map_path)
        self._width = self._map_info["width"]
        self._height = self._map_info["height"]
        self._max_time = self._map_info["max_time"]

        self.action_space = spaces.Discrete(len(self._ACTION_LOOKUP))
        obs_len = len(self._OBS_CHANNELS) * self._width * self._height + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )

        self.world: Optional[RealWorld] = None
        self.character: Optional[RLCharacter] = None
        self._last_score: float = 0.0

        # Exit progress shaping parameters
        self._prev_exit_decay: float = 0.0
        self._exit_decay_rate: float = 0.35  # controls exponential falloff
        self._exit_progress_base: float = 12.0  # medium baseline reward

        # Event-driven reward magnitudes
        self._wall_destroy_reward: float = 6.0  # low
        self._monster_kill_reward: float = 150.0  # high
        self._self_bomb_penalty: float = 600.0  # very high penalty

        # Additional outcome shaping
        self._step_penalty: float = 0.2
        self._death_penalty: float = 200.0
        self._exit_completion_reward: float = 300.0

    # ---------------------------------------------------------------------
    # Gym / Gymnasium API
    # ---------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        # Build a fresh world instance for every episode
        self.world = self._build_world()
        self.character = RLCharacter("sb3-agent", "C", 0, 0)
        self.character.reset_flags()
        self.world.add_character(self.character)
        # Monsters as defined in variant5.py
        self.world.add_monster(StupidMonster("stupid", "S", 3, 5))
        self.world.add_monster(SelfPreservingMonster("aggressive", "A", 3, 13, 2))
        self._last_score = self.world.scores[self.character.name]

        # Initialize rolling state
        dist_to_exit = self._manhattan_distance_to_exit()
        self._prev_exit_decay = self._compute_exit_decay(dist_to_exit)

        observation = self._collect_observation()
        info: Dict[str, float] = {"score": self._last_score}

        if _USES_GYMNASIUM_API:
            return observation, info
        return observation

    def step(self, action: int):
        if self.world is None or self.character is None:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected integer in action space")

        dx, dy, place_bomb = self._ACTION_LOOKUP[action]
        self.character.move(dx, dy)
        if place_bomb:
            self.character.place_bomb()

        # Advance the simulation one tick
        self.world.next()
        events = list(self.world.events)

        terminated, success = self._compute_termination_flags(events)
        truncated = bool(self.world.time <= 0 and not terminated)

        # Update scores-based reward signal
        current_score = self.world.scores.get(self.character.name, self._last_score)
        reward = -self._step_penalty

        # Exit-seeking reward: exponential bonus for reducing distance to exit
        dist_to_exit = self._manhattan_distance_to_exit()
        current_exit_decay = self._compute_exit_decay(dist_to_exit)
        reward += self._exit_progress_base * (current_exit_decay - self._prev_exit_decay)
        self._prev_exit_decay = current_exit_decay

        for ev in events:
            if ev.tpe == Event.BOMB_HIT_WALL and ev.character is self.character:
                reward += self._wall_destroy_reward
            elif ev.tpe == Event.BOMB_HIT_MONSTER and ev.character is self.character:
                reward += self._monster_kill_reward
            elif ev.tpe == Event.BOMB_HIT_CHARACTER and ev.other is self.character:
                reward -= self._self_bomb_penalty
            elif ev.tpe == Event.CHARACTER_FOUND_EXIT and ev.character is self.character:
                reward += self._exit_completion_reward

        if terminated and not success:
            reward -= self._death_penalty

        self._last_score = current_score

        # Prepare next-step decisions for all entities
        self.world.next_decisions()

        observation = self._collect_observation()
        info: Dict[str, object] = {
            "events": events,
            "score": current_score,
            "success": success,
        }

        if _USES_GYMNASIUM_API:
            return observation, float(reward), terminated, truncated, info

        done = terminated or truncated
        return observation, float(reward), done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_world(self) -> RealWorld:
        world = RealWorld.from_params(
            self._width,
            self._height,
            self._map_info["max_time"],
            self._map_info["bomb_time"],
            self._map_info["expl_duration"],
            self._map_info["expl_range"],
        )
        for y, row in enumerate(self._map_info["grid" ]):
            for x, cell in enumerate(row):
                if cell == "E":
                    world.add_exit(x, y)
                elif cell == "W":
                    world.add_wall(x, y)
        return world

    def _collect_observation(self) -> np.ndarray:
        assert self.world is not None and self.character is not None
        grids: Dict[str, np.ndarray] = {
            channel: np.zeros((self._height, self._width), dtype=np.float32)
            for channel in self._OBS_CHANNELS
        }
        for x in range(self._width):
            for y in range(self._height):
                if self.world.wall_at(x, y):
                    grids["walls"][y, x] = 1.0
                if self.world.exit_at(x, y):
                    grids["exit"][y, x] = 1.0
                if self.world.bomb_at(x, y):
                    grids["bombs"][y, x] = 1.0
                if self.world.explosion_at(x, y):
                    grids["explosions"][y, x] = 1.0
                if self.world.monsters_at(x, y):
                    grids["monsters"][y, x] = 1.0
                if self.world.characters_at(x, y):
                    for ch in self.world.characters_at(x, y):
                        if ch.name == self.character.name:
                            grids["agent"][y, x] = 1.0
        time_feature = np.array([self.world.time / float(self._max_time)], dtype=np.float32)
        score_feature = np.array([
            self.world.scores.get(self.character.name, self._last_score) / 200.0
        ], dtype=np.float32)
        flat_channels = [grids[channel].ravel() for channel in self._OBS_CHANNELS]
        return np.concatenate(flat_channels + [time_feature, score_feature]).astype(np.float32)

    def _compute_termination_flags(self, events: Iterable[Event]) -> Tuple[bool, bool]:
        assert self.character is not None
        terminated = False
        success = False
        for ev in events:
            if ev.tpe == Event.CHARACTER_FOUND_EXIT and ev.character is self.character:
                terminated = True
                success = True
            elif ev.tpe == Event.CHARACTER_KILLED_BY_MONSTER and ev.character is self.character:
                terminated = True
            elif ev.tpe == Event.BOMB_HIT_CHARACTER and ev.other is self.character:
                terminated = True
        if not terminated:
            # Check whether the agent has disappeared from the world grid
            if not self._agent_still_present():
                terminated = True
        return terminated, success

    def _agent_still_present(self) -> bool:
        assert self.world is not None and self.character is not None
        for chars in self.world.characters.values():
            if self.character in chars:
                return True
        return False

    def render(self):
        if self.render_mode == "console" and self.world is not None:
            self.world.printit()
        else:
            raise NotImplementedError("Only console rendering is supported.")

    def close(self):
        self.world = None
        self.character = None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_map(path: Path) -> Dict[str, object]:
        with path.open("r", encoding="ascii") as fd:
            max_time = int(fd.readline().split()[1])
            bomb_time = int(fd.readline().split()[1])
            expl_duration = int(fd.readline().split()[1])
            expl_range = int(fd.readline().split()[1])
            border_line = fd.readline()
            width = len(border_line.strip()) - 2
            rows: List[str] = []
            row = fd.readline()
            while row and row.startswith("|"):
                content = row.strip()[1:-1]
                if len(content) != width:
                    raise RuntimeError("Invalid map row length", len(content), width)
                rows.append(content)
                row = fd.readline()
        height = len(rows)
        return {
            "width": width,
            "height": height,
            "max_time": max_time,
            "bomb_time": bomb_time,
            "expl_duration": expl_duration,
            "expl_range": expl_range,
            "grid": rows,
        }

    def _compute_exit_decay(self, distance: Optional[int]) -> float:
        if distance is None:
            return 0.0
        return math.exp(-self._exit_decay_rate * float(distance))

    def _manhattan_distance_to_exit(self) -> Optional[int]:
        if self.world is None or self.character is None or not self.world.exitcell:
            return None
        ex, ey = self.world.exitcell
        return int(abs(ex - self.character.x) + abs(ey - self.character.y))

__all__ = ["BombermanVariant5Env", "RLCharacter"]
