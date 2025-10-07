import os
import sys
from typing import Optional, Tuple, Dict, Any, List, Set
import random

import numpy as np

# Engine imports (resolve relative to this file)
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_ENGINE_DIR = os.path.join(_BASE_DIR, 'Bomberman')
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)
from real_world import RealWorld  # noqa: E402
from sensed_world import SensedWorld  # noqa: E402
from entity import CharacterEntity  # noqa: E402
from events import Event  # noqa: E402

# RL
import gymnasium as gym
from gymnasium import spaces

from .utils import parse_map_to_world, build_observation, observation_size, num_actions, decode_action


class RLCharacter(CharacterEntity):
    """Character controlled by external actions via a buffer set by the environment."""

    def __init__(self, name: str, avatar: str, x: int, y: int):
        super().__init__(name, avatar, x, y)
        self._pending_action: Optional[int] = None

    def set_action(self, action_index: int) -> None:
        self._pending_action = action_index

    def do(self, wrld: Optional[SensedWorld]):
        if self._pending_action is None:
            # No-op if no action set
            self.move(0, 0)
            return
        dx, dy, bomb = decode_action(self._pending_action)
        self.move(dx, dy)
        if bomb:
            self.place_bomb()
        # Consume action
        self._pending_action = None


class BombermanProject2Env(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(self, map_path: str = os.path.join(os.path.dirname(__file__), 'map.txt'),
                 render_mode: Optional[str] = None,
                 bottom_start: bool = True,
                 randomize_bottom_start: bool = True,
                 simple_reward: bool = True):
        super().__init__()
        self.map_path = map_path
        self.render_mode = render_mode
        self.bottom_start = bottom_start
        self.randomize_bottom_start = randomize_bottom_start
        self.simple_reward = simple_reward

        # World template used to infer sizes
        tmp_world = parse_map_to_world(self.map_path)
        self._obs_size = observation_size(tmp_world)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self._obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(num_actions())

        self.world: Optional[RealWorld] = None
        self.me: Optional[RLCharacter] = None
        self._last_score: int = 0
        self._last_pos: Optional[Tuple[int, int]] = None
        self._last_dist: Optional[float] = None
        self._wall_rows: List[int] = []
        self._passed_rows: Set[int] = set()
        self._last_action_tuple: Tuple[int, int, int] = (0, 0, 0)
        self._best_y_reached: Optional[int] = None

    def _spawn_world(self) -> None:
        self.world = parse_map_to_world(self.map_path)
        # Determine start position
        if not self.bottom_start:
            # Place agent at (0,0) unless blocked; otherwise find first empty cell scanning rows
            start_x, start_y = 0, 0
            if self.world.wall_at(start_x, start_y):
                placed = False
                for y in range(self.world.height()):
                    for x in range(self.world.width()):
                        if self.world.empty_at(x, y):
                            start_x, start_y = x, y
                            placed = True
                            break
                    if placed:
                        break
        else:
            # Compute bottom section minimum y as one row below the lowest full-width wall
            wall_rows: List[int] = []
            for y in range(self.world.height()):
                full_wall = True
                for x in range(self.world.width()):
                    if not self.world.wall_at(x, y):
                        full_wall = False
                        break
                if full_wall:
                    wall_rows.append(y)
            min_bottom_y = (max(wall_rows) + 1) if wall_rows else 0
            # Collect empty cells in bottom section
            candidates: List[Tuple[int, int]] = []
            for y in range(min_bottom_y, self.world.height()):
                for x in range(self.world.width()):
                    if self.world.empty_at(x, y):
                        candidates.append((x, y))
            if not candidates:
                start_x, start_y = 0, 0
            else:
                if self.randomize_bottom_start:
                    start_x, start_y = random.choice(candidates)
                else:
                    # Pick deterministic left-most, lowest
                    start_x, start_y = min(candidates, key=lambda p: (p[0], -p[1]))
        self.me = RLCharacter('me', 'C', start_x, start_y)
        self.world.add_character(self.me)
        # Snapshot score
        self._last_score = self.world.scores.get(self.me.name, 0)
        # Track last position and distance to exit for shaping
        self._last_pos = (self.me.x, self.me.y)
        self._best_y_reached = self.me.y
        if self.world.exitcell is not None:
            ex, ey = self.world.exitcell
            self._last_dist = abs(self.me.x - ex) + abs(self.me.y - ey)
        else:
            self._last_dist = None
        # Precompute horizontal wall rows (full-width walls) for milestone rewards
        self._wall_rows = []
        for y in range(self.world.height()):
            full_wall = True
            for x in range(self.world.width()):
                if not self.world.wall_at(x, y):
                    full_wall = False
                    break
            if full_wall:
                self._wall_rows.append(y)
        self._passed_rows = set()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._spawn_world()
        obs = build_observation(self.world, self.me.name, self._last_action_tuple)
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int):
        assert self.world is not None and self.me is not None

        # No bomb placement in simplified navigation training

        # Set action for this tick
        self.me.set_action(int(action))
        # Apply decisions then step world
        self.world.next_decisions()
        self.world, events = self.world.next()

        # Observation
        obs = build_observation(self.world, self.me.name, self._last_action_tuple)

        # Minimal reward: based solely on progress towards exit
        reward = 0.0

        # Termination: time over or agent removed from world
        terminated = False
        truncated = False
        # Agent alive?
        alive = False
        for _, clist in self.world.characters.items():
            for c in clist:
                if c.name == self.me.name:
                    alive = True
                    break
            if alive:
                break
        if not alive:
            terminated = True
        if self.world.time <= 0:
            truncated = True

        # Progress towards exit (outcome-based)
        # current position
        cur_pos = None
        for _, clist in self.world.characters.items():
            for c in clist:
                if c.name == self.me.name:
                    cur_pos = (c.x, c.y)
                    break
            if cur_pos is not None:
                break
        prev_pos = self._last_pos
        self._last_pos = cur_pos

        # distance shaping towards exit (primary signal)
        if self.world.exitcell is not None and cur_pos is not None:
            ex, ey = self.world.exitcell
            cur_dist = abs(cur_pos[0] - ex) + abs(cur_pos[1] - ey)
            if self._last_dist is not None:
                reward += 2.0 * (self._last_dist - cur_dist)
            self._last_dist = cur_dist

        # One-time depth milestone: reward new best (lowest) y reached this run (optional)
        if not self.simple_reward:
            if cur_pos is not None and self._best_y_reached is not None:
                cy = cur_pos[1]
                if cy > self._best_y_reached:
                    height = self.world.height()
                    depth_norm = float(cy + 1) / float(max(1, height))
                    depth_bonus = 8.0 * (1.0 + 2.0 * depth_norm)
                    reward += depth_bonus
                    self._best_y_reached = cy

        # No per-step offset, no extra wall bonuses: keep schema minimal

        # No death penalty in safe simplified navigation

        # Update last action tuple (dx,dy,bomb=0 as bombs are auto)
        dx, dy, _ = decode_action(int(action))
        self._last_action_tuple = (dx, dy, 0)

        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'ansi' and self.world is not None:
            # Text rendering delegate to world
            self.world.printit()


