import sys
import os
from typing import Tuple

import numpy as np

# Ensure engine is importable relative to repo root
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_ENGINE_DIR = os.path.join(_BASE_DIR, 'Bomberman')
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from real_world import RealWorld  # noqa: E402


# Movement primitives: 8 directions + stay
MOVE_DELTAS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),  (0, 0),  (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
)


def num_actions() -> int:
    # Movement only; bomb placement is handled externally by the environment
    return len(MOVE_DELTAS)


def decode_action(action_index: int) -> Tuple[int, int, bool]:
    move_index = int(action_index) % len(MOVE_DELTAS)
    dx, dy = MOVE_DELTAS[move_index]
    return dx, dy, False


def parse_map_to_world(map_path: str) -> RealWorld:
    with open(map_path, 'r') as fd:
        max_time = int(fd.readline().split()[1])
        bomb_time = int(fd.readline().split()[1])
        expl_duration = int(fd.readline().split()[1])
        expl_range = int(fd.readline().split()[1])
        width = len(fd.readline()) - 3
        startpos = fd.tell()
        height = 0
        row = fd.readline()
        while row and row[0] == '|':
            height += 1
            if len(row) != width + 3:
                raise RuntimeError("Row", height, "is not", width, "characters long")
            row = fd.readline()
    world = RealWorld.from_params(width, height, max_time, bomb_time, expl_duration, expl_range)
    with open(map_path, 'r') as fd:
        # skip headers
        for _ in range(4):
            fd.readline()
        # top border consumed in width computation, skip it
        fd.readline()
        for y in range(0, height):
            line = fd.readline()
            for x in range(0, width):
                cell = line[x + 1]
                if cell == 'E':
                    if not world.exitcell:
                        world.add_exit(x, y)
                    else:
                        raise RuntimeError("There can be only one exit cell, first one found already")
                elif cell == 'W':
                    world.add_wall(x, y)
    return world


_EXTRA_FEATURES = 5  # dx_to_exit, dy_to_exit, last_dx, last_dy, last_bomb

def build_observation(world, me_name: str, last_action: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    width = world.width()
    height = world.height()
    channels = 6  # wall, exit, bomb, explosion, player, unsafe_next
    obs = np.zeros((height, width, channels), dtype=np.float32)

    # Static layers
    for x in range(width):
        for y in range(height):
            if world.wall_at(x, y):
                obs[y, x, 0] = 1.0
            if world.exit_at(x, y):
                obs[y, x, 1] = 1.0

    # Bombs and explosions (normalized timers)
    for idx, bomb in world.bombs.items():
        x = bomb.x
        y = bomb.y
        denom = max(1, world.bomb_time)
        obs[y, x, 2] = max(0.0, min(1.0, float(bomb.timer) / float(denom)))

    for idx, expl in world.explosions.items():
        x = expl.x
        y = expl.y
        denom = max(1, world.expl_duration)
        obs[y, x, 3] = max(0.0, min(1.0, float(expl.timer) / float(denom)))

    # Predict unsafe tiles for the next step:
    # 1) Existing explosions that will persist next tick (timer > 0 now)
    for idx, expl in world.explosions.items():
        if expl.timer > 0:
            obs[expl.y, expl.x, 5] = 1.0
    # 2) Bombs that will explode next tick (timer == 0 now)
    def mark_predicted_blast(bx: int, by: int):
        # mark center
        obs[by, bx, 5] = 1.0
        # four directions
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            xx, yy = bx + dx, by + dy
            r = 0
            while (r < world.expl_range and 0 <= xx < width and 0 <= yy < height):
                # cannot pass exit or another bomb
                if world.exit_at(xx, yy) or world.bomb_at(xx, yy):
                    break
                obs[yy, xx, 5] = 1.0
                # stop at first wall (included as unsafe)
                if world.wall_at(xx, yy):
                    break
                xx += dx
                yy += dy
                r += 1

    for idx, bomb in world.bombs.items():
        if bomb.timer == 0:
            mark_predicted_blast(bomb.x, bomb.y)

    # Player position one-hot
    me_x, me_y, me_dx, me_dy = 0, 0, 0, 0
    for _, clist in world.characters.items():
        for c in clist:
            if c.name == me_name:
                obs[c.y, c.x, 4] = 1.0
                me_x, me_y, me_dx, me_dy = c.x, c.y, c.dx, c.dy

    flat = obs.reshape(-1)

    # Append extra global features: vector to exit and last action (dx,dy,bomb)
    extra = np.zeros((_EXTRA_FEATURES,), dtype=np.float32)
    # dx,dy to exit normalized
    if world.exitcell is not None:
        ex, ey = world.exitcell
        extra[0] = float(ex - me_x) / float(max(1, width))
        extra[1] = float(ey - me_y) / float(max(1, height))
    # last action
    ldx, ldy, lb = last_action
    extra[2] = float(ldx)
    extra[3] = float(ldy)
    extra[4] = float(1.0 if lb else 0.0)

    return np.concatenate([flat, extra], axis=0)


def observation_size(world) -> int:
    return world.width() * world.height() * 6 + _EXTRA_FEATURES


