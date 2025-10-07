import sys
import random
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

from game import Game

sys.path.insert(1, '../teamNN')
from character1 import Character1


# Create the game
g = Game.fromfile('map.txt')

# Spawn within fixed bottom rows
target_rows = [15, 16, 17]
candidates = []
for y in target_rows:
    if 0 <= y < g.world.height():
        for x in range(g.world.width()):
            if g.world.empty_at(x, y):
                candidates.append((x, y))
if candidates:
    start_x, start_y = random.choice(candidates)
else:
    start_x, start_y = 0, 0

# Place character at computed bottom-section start
g.add_character(Character1("me", # name
                           "C",  # avatar
                           start_x, start_y,  # position
                           model_path='models/best_model.zip'))

# Run!
g.go(0)


