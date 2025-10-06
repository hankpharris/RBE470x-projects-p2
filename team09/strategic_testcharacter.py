"""Rule-based Bomberman agent for variant 5."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / "Bomberman"))
sys.path.insert(1, str(_THIS_DIR))

from entity import CharacterEntity  # noqa: E402


Coord = Tuple[int, int]
Move = Tuple[int, int]


class TestCharacter(CharacterEntity):
    """Deterministic agent that clears a path to the exit safely."""

    MOVE_ORDER: Sequence[Move] = ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))

    def __init__(self, name: str, avatar: str, x: int, y: int):
        super().__init__(name, avatar, x, y)
        self._escape_plan: List[Move] = []
        self._bomb_active: bool = False

    def do(self, wrld) -> None:  # noqa: D401
        if wrld is None:
            return
        me = wrld.me(self)
        if me is None:
            return

        bombs = list(getattr(wrld, "bombs", {}).values())
        danger_tiles = self._compute_danger_tiles(wrld, bombs)
        monsters = self._collect_monster_tiles(wrld)

        if self._escape_plan:
            move = self._next_escape_step(wrld, me, danger_tiles)
            if move is not None:
                self._apply_move(move)
                return

        if (me.x, me.y) in danger_tiles:
            safe = self._select_safe_move(wrld, me, danger_tiles, monsters)
            self._apply_move(safe or (0, 0))
            return

        target_step = self._step_toward_exit(wrld, me, danger_tiles, monsters)
        if target_step is not None:
            self._apply_move(target_step)
            return

        fallback = self._select_safe_move(wrld, me, danger_tiles, monsters)
        self._apply_move(fallback or (0, 0))

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def _apply_move(self, move: Move) -> None:
        dx, dy = move
        self.move(dx, dy)

    def _compute_danger_tiles(self, wrld, bombs: Iterable) -> set[Coord]:
        danger: set[Coord] = set()
        for expl in getattr(wrld, "explosions", {}).values():
            danger.add((expl.x, expl.y))
        for bomb in bombs:
            tiles = self._blast_tiles(wrld, bomb)
            danger.update(tiles)
        return danger

    def _collect_monster_tiles(self, wrld) -> set[Coord]:
        tiles: set[Coord] = set()
        for monsters in wrld.monsters.values():
            for monster in monsters:
                tiles.add((monster.x, monster.y))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx = monster.x + dx
                        ny = monster.y + dy
                        if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                            tiles.add((nx, ny))
        return tiles

    def _blast_tiles(self, wrld, bomb) -> set[Coord]:
        tiles = {(bomb.x, bomb.y)}
        width = wrld.width()
        height = wrld.height()
        rng = wrld.expl_range
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            xx, yy = bomb.x, bomb.y
            for _ in range(rng):
                xx += dx
                yy += dy
                if xx < 0 or xx >= width or yy < 0 or yy >= height:
                    break
                tiles.add((xx, yy))
                if wrld.wall_at(xx, yy) or wrld.bomb_at(xx, yy):
                    break
        return tiles

    def _next_escape_step(self, wrld, me, danger_tiles: set[Coord]) -> Optional[Move]:
        while self._escape_plan:
            move = self._escape_plan.pop(0)
            nx = me.x + move[0]
            ny = me.y + move[1]
            if not self._cell_blocked(wrld, nx, ny) and (nx, ny) not in danger_tiles:
                self._bomb_active = False
                return move
        self._bomb_active = False
        return None

    def _step_toward_exit(self, wrld, me, danger_tiles: set[Coord], monsters: set[Coord]) -> Optional[Move]:
        if not wrld.exitcell:
            return None
        ex, ey = wrld.exitcell
        dx = 0 if me.x == ex else (1 if ex > me.x else -1)
        dy = 0 if me.y == ey else (1 if ey > me.y else -1)

        preferred_moves: List[Move] = []
        if dx != 0:
            preferred_moves.append((dx, 0))
        if dy != 0:
            preferred_moves.append((0, dy))
        for move in preferred_moves:
            nx = me.x + move[0]
            ny = me.y + move[1]
            if not self._in_bounds(wrld, nx, ny):
                continue
            if self._cell_blocked(wrld, nx, ny):
                if self._attempt_bomb(wrld, me, move, danger_tiles, monsters):
                    return None
                continue
            if (nx, ny) in danger_tiles or (nx, ny) in monsters:
                continue
            return move

        candidate = self._select_safe_move(wrld, me, danger_tiles, monsters, bias_exit=True)
        if candidate is not None:
            return candidate
        return None

    def _attempt_bomb(self, wrld, me, move: Move, danger_tiles: set[Coord], monsters: set[Coord]) -> bool:
        if self._bomb_active:
            return False
        if self._has_active_bomb(wrld):
            return False
        nx = me.x + move[0]
        ny = me.y + move[1]
        if not self._in_bounds(wrld, nx, ny):
            return False
        if not (wrld.wall_at(nx, ny) or wrld.monsters_at(nx, ny)):
            return False
        # Place bomb and schedule retreat
        self.place_bomb()
        self._bomb_active = True
        blast_zone = self._hypothetical_blast(wrld, me.x, me.y)
        expanded_danger = danger_tiles | blast_zone
        self._escape_plan = self._plan_escape_route(wrld, me, expanded_danger, monsters)
        return True

    def _plan_escape_route(
        self,
        wrld,
        me,
        danger_tiles: set[Coord],
        monsters: set[Coord],
    ) -> List[Move]:
        safe_moves: List[Move] = []
        for move in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx = me.x + move[0]
            ny = me.y + move[1]
            if not self._in_bounds(wrld, nx, ny):
                continue
            if self._cell_blocked(wrld, nx, ny):
                continue
            if (nx, ny) in danger_tiles or (nx, ny) in monsters:
                continue
            safe_moves.append(move)
        if not safe_moves:
            return []
        return list(safe_moves[:3])

    def _select_safe_move(
        self,
        wrld,
        me,
        danger_tiles: set[Coord],
        monsters: set[Coord],
        bias_exit: bool = False,
    ) -> Optional[Move]:
        candidates: List[Tuple[float, Move]] = []
        for move in self.MOVE_ORDER:
            nx = me.x + move[0]
            ny = me.y + move[1]
            if not self._in_bounds(wrld, nx, ny):
                continue
            if self._cell_blocked(wrld, nx, ny):
                continue
            if (nx, ny) in danger_tiles or (nx, ny) in monsters:
                continue
            score = 0.0
            if bias_exit and wrld.exitcell:
                ex, ey = wrld.exitcell
                score = abs(nx - ex) + abs(ny - ey)
            candidates.append((score, move))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _in_bounds(self, wrld, x: int, y: int) -> bool:
        return 0 <= x < wrld.width() and 0 <= y < wrld.height()

    def _cell_blocked(self, wrld, x: int, y: int) -> bool:
        return wrld.wall_at(x, y) or bool(wrld.monsters_at(x, y)) or bool(wrld.bomb_at(x, y))

    def _has_active_bomb(self, wrld) -> bool:
        for bomb in wrld.bombs.values():
            if bomb.owner == wrld.me(self):
                return True
        return False

    def _hypothetical_blast(self, wrld, x: int, y: int) -> set[Coord]:
        class DummyBomb:
            def __init__(self, bx: int, by: int):
                self.x = bx
                self.y = by

        return self._blast_tiles(wrld, DummyBomb(x, y))


__all__ = ["TestCharacter"]
