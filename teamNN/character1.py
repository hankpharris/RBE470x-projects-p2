import os
import sys
from typing import Optional

# Engine path relative to repo root
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_ENGINE_DIR = os.path.join(_BASE_DIR, 'Bomberman')
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from entity import CharacterEntity  # noqa: E402
from sensed_world import SensedWorld  # noqa: E402

from stable_baselines3 import DQN

from project2.utils import build_observation, decode_action  # noqa: E402


class Character1(CharacterEntity):
    def __init__(self, name, avatar, x, y, model_path: Optional[str] = None):
        super().__init__(name, avatar, x, y)
        self.model_path = model_path
        self.model: Optional[DQN] = None

    def _ensure_model(self):
        if (self.model is None) and self.model_path and os.path.exists(self.model_path):
            self.model = DQN.load(self.model_path)

    def do(self, wrld: Optional[SensedWorld]):
        if wrld is None or self.model_path is None:
            self.move(0, 0)
            return
        self._ensure_model()
        if self.model is None:
            self.move(0, 0)
            return
        # Auto-bomb: place a bomb if it would hit a wall and no bomb is active for me
        def has_active_bomb() -> bool:
            for _, b in wrld.bombs.items():
                if b.owner.name == self.name:
                    return True
            return False

        def will_hit_wall() -> bool:
            x0, y0 = wrld.me(self).x, wrld.me(self).y
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                xx, yy = x0 + dx, y0 + dy
                r = 0
                while (r < wrld.expl_range and 0 <= xx < wrld.width() and 0 <= yy < wrld.height()):
                    if wrld.exit_at(xx, yy) or wrld.bomb_at(xx, yy):
                        break
                    if wrld.wall_at(xx, yy):
                        return True
                    xx += dx
                    yy += dy
                    r += 1
            return False

        if not has_active_bomb() and will_hit_wall():
            self.place_bomb()

        obs = build_observation(wrld, self.name)
        action, _ = self.model.predict(obs, deterministic=True)
        dx, dy, bomb = decode_action(int(action))
        self.move(dx, dy)
        # Bombs are auto-placed; ignore model bomb flag


