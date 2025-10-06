# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from character1 import Character1


# Create the game
g = Game.fromfile('map.txt')

# TODO Add your character
# g.add_character(Character1("me", # name
#                            "C",  # avatar
#                            0, 0,  # position
#                            model_path='models/best_model.zip'  # model
# ))

from character2 import Character2

g.add_character(Character2("me", # name
                           "C",  # avatar
                           0, 0  # position
))


# Run!
g.go(0)
