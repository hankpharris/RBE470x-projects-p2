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
        obs = build_observation(wrld, self.name)
        action, _ = self.model.predict(obs, deterministic=True)
        dx, dy, bomb = decode_action(int(action))
        self.move(dx, dy)
        # Ignore any bomb flag; bombs are disabled in simplified navigation


