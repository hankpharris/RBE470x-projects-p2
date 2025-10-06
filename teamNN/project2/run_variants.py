import os
import sys
import argparse
import random

# Resolve repository root and add engine/team to sys.path
_HERE = os.path.abspath(os.path.dirname(__file__))
_BASE_DIR = os.path.abspath(os.path.join(_HERE, '..', '..'))
_ENGINE_DIR = os.path.join(_BASE_DIR, 'Bomberman')
_TEAM_DIR = os.path.join(_BASE_DIR, 'teamNN')
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)
if _TEAM_DIR not in sys.path:
    sys.path.insert(1, _TEAM_DIR)

from game import Game
from events import Event
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

# Characters (from this team repo)
from character2 import Character2
from character3 import Character3


def build_game_for_variant(variant):
    # Recreate the setup from the variant files, but without auto-running
    # All variants use the same map
	map_path = os.path.join(_HERE, 'map.txt')
	g = Game.fromfile(map_path)
	
	if variant == 1:
		# Variant 1 (project 2): Alone in the world
		g.add_character(Character2("me", "C", 0, 0))
		return g
	elif variant == 2:
		# Variant 2 (project 2): Stupid monster
		g.add_monster(StupidMonster("stupid", "S", 3, 9))
		g.add_character(Character2("me", "C", 0, 0))
		return g
	elif variant == 3:
		# Variant 3 (project 2): Self-preserving monster
		g.add_monster(SelfPreservingMonster("selfpreserving", "S", 3, 9, 1))
		g.add_character(Character2("me", "C", 0, 0))
		return g
	elif variant == 4:
		# Variant 4 (project 2): Aggressive monster
		g.add_monster(SelfPreservingMonster("aggressive", "A", 3, 13, 2))
		g.add_character(Character3("me", "C", 0, 0))
		return g
	elif variant == 5:
		# Variant 5 (project 2): Stupid and Aggressive monsters together
		g.add_monster(StupidMonster("stupid", "S", 3, 5))
		g.add_monster(SelfPreservingMonster("aggressive", "A", 3, 13, 2))
		g.add_character(Character3("me", "C", 0, 0))
		return g
	else:
		raise ValueError("Unknown variant: {}".format(variant))


def run_episode(variant, wait_ms=1):
	# Small randomization across episodes unless user wants determinism outside
	random.seed()
	g = build_game_for_variant(variant)
	# Run until done, track success from events
	success = False
	def _step():
		# match Game.go(>0) behavior but inline to capture events
		import pygame
		pygame.time.wait(abs(wait_ms))
		return
	
	# Initialize a minimal render like Game.go does
	from colorama import init as colorama_init, deinit as colorama_deinit
	import pygame
	colorama_init(autoreset=True)
	g.display_gui()
	g.draw()
	_step()
	while not g.done():
		(g.world, g.events) = g.world.next()
		# Detect success this tick
		for e in g.events:
			if e.tpe == Event.CHARACTER_FOUND_EXIT:
				success = True
				break
		g.display_gui()
		g.draw()
		_step()
		g.world.next_decisions()
	colorama_deinit()
	return success


def main():
	parser = argparse.ArgumentParser(description="Run Bomberman variants repeatedly and report success rate")
	parser.add_argument("--variant", type=int, required=True, choices=[1,2,3,4,5], help="Variant number to run (1-5)")
	parser.add_argument("--count", type=int, default=10, help="Number of episodes to run")
	parser.add_argument("--headless", action="store_true", help="Run without opening a window for speed")
	parser.add_argument("--wait-ms", type=int, default=1, help="Milliseconds to wait per tick (>=0)")
	args = parser.parse_args()

	if args.headless:
		# Use SDL dummy video driver to avoid opening a window
		os.environ["SDL_VIDEODRIVER"] = "dummy"
		os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


	import pygame  
	total = args.count
	successes = 0
	for _ in range(total):
		ok = run_episode(args.variant, wait_ms=max(0, args.wait_ms))
		if ok:
			successes += 1

	rate = (successes / float(total)) if total > 0 else 0.0
	print("variant={}, count={}, successes={}, success_rate={:.2%}".format(args.variant, total, successes, rate))


if __name__ == "__main__":
	main()



