"""Character5: Based on Character4 but switches to DQN only in bottom rows.

This character mirrors Character4 (flee logic, A*, south fallback, etc.).
Additionally, once the agent is in the training band rows y ∈ {15,16,17}, it
uses the trained DQN model (movement-only) to pick the next action for that
tick, provided it is not in the flee state. Outside that band, or while
fleeing, it behaves exactly like Character4.
"""

import sys
import heapq
from collections import deque
import math
import os
from typing import Optional

#This is the start of our code, algorithmically controlling behavior
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

# DQN inference
from stable_baselines3 import DQN
from project2.utils import build_observation, decode_action


class Character5(CharacterEntity):

	def __init__(self, name, avatar, x, y, model_path: Optional[str] = None):
		super().__init__(name, avatar, x, y)
		self.path = []
		self.path_index = 0
		self.fleeing = False
		self.placed_bomb_this_flee = False
		self.owned_bomb_pos = None
		self.post_blast_cooldown = 0  # turns of extra danger row/col after bomb disappears
		# DQN model path (defaults to teamNN/project2/models/best_model.zip)
		self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'project2', 'models', 'best_model.zip')
		self.model: Optional[DQN] = None

	def _ensure_model(self):
		if self.model is None and self.model_path and os.path.exists(self.model_path):
			self.model = DQN.load(self.model_path)

	def do(self, wrld):
		# Gather monsters and exit
		exit_pos = self.find_exit(wrld)
		monsters = self.find_monsters(wrld)

		# Track our bomb state across turns (detect detonation and run cooldown)
		self._update_bomb_state(wrld)

		# Decide if we should be fleeing based on Chebyshev distance (diagonal counts as 1)
		min_monster_dist = self._min_chebyshev_distance((self.x, self.y), monsters) if monsters else float('inf')
		race_advantage = self._has_exit_race_advantage(wrld, exit_pos, monsters)
		entering_flee_now = (not self.fleeing) and (min_monster_dist <= 5) and (not race_advantage)
		if entering_flee_now:
			self.fleeing = True
			self.placed_bomb_this_flee = False

		# Place bomb immediately upon entering flee state (can still move this turn)
		if entering_flee_now and not self.placed_bomb_this_flee:
			self.place_bomb()
			self.placed_bomb_this_flee = True
			self.owned_bomb_pos = (self.x, self.y)

		# Choose action; remain in fleeing as long as any bomb exists or our bomb effects are active
		active_flee = self.fleeing or self._bomb_effects_active(wrld) or self._any_bomb_exists(wrld)
		if active_flee:
			self.fleeing = True
			next_pos = self._choose_flee_move(wrld, exit_pos, monsters)
		else:
			# If inside bottom training band, prefer DQN guidance for navigation
			if self._in_bottom_band(wrld):
				self._ensure_model()
				if self.model is not None:
					obs = build_observation(wrld, self.name)
					action, _ = self.model.predict(obs, deterministic=True)
					dx, dy, bomb = decode_action(int(action))
					print(f"[Character5] approach=MODEL_BOTTOM cur=({self.x},{self.y}) next=({self.x+dx},{self.y+dy}) monsters={monsters}")
					self.move(dx, dy)
					return
			# Otherwise, fall back to Character4's base navigation

			# Base A* movement to exit
			next_pos = None
			if exit_pos is not None:
				path = self.astar_pathfinding(wrld, (self.x, self.y), exit_pos)
				if path:
					next_pos = path[0]
				else:
					# No path to exit due to blocked sections: greedy south fallback.
					# Try to move south as far as possible (prioritize S, then SE, then SW).
					south_move = self._south_fallback_move(wrld)
					if south_move is not None:
						next_pos = south_move
					else:
						# We cannot progress further south; place a bomb and enter flee mode immediately.
						self.place_bomb()
						self.fleeing = True
						self.placed_bomb_this_flee = True
						self.owned_bomb_pos = (self.x, self.y)
						# Pick a flee move right away for this tick
						next_pos = self._choose_flee_move(wrld, exit_pos, monsters)

		# Exit fleeing ONLY when our bomb's explosion window has fully cleared
		if self.fleeing and not self._bomb_effects_active(wrld):
			self.fleeing = False
			self.placed_bomb_this_flee = False
			self.owned_bomb_pos = None

		# Default to staying if no move found
		if next_pos is None:
			next_pos = (self.x, self.y)

		# Compute delta and move
		dx = max(-1, min(1, next_pos[0] - self.x))
		dy = max(-1, min(1, next_pos[1] - self.y))
		# Log required info
		approach = 'FLEEING' if active_flee else 'BASE'
		print(f"[Character5] approach={approach} cur=({self.x},{self.y}) next=({self.x+dx},{self.y+dy}) monsters={monsters}")
		self.move(dx, dy)

	def _in_bottom_band(self, wrld):
		# Switch to DQN only within the bottom training band rows
		return self.y in (15, 16, 17)

	def _south_fallback_move(self, wrld):
		# Prefer straight south, then diagonals (SE, SW) if safe and within bounds
		w, h = wrld.width(), wrld.height()
		candidates = [(0, 1), (1, 1), (-1, 1)]
		for dx, dy in candidates:
			nx, ny = self.x + dx, self.y + dy
			if 0 <= nx < w and 0 <= ny < h:
				if (not wrld.wall_at(nx, ny)
					and not wrld.explosion_at(nx, ny)
					and not wrld.bomb_at(nx, ny)
					and not self._imminent_blast_at(nx, ny, wrld)):
					return (nx, ny)
		return None

	def _has_exit_race_advantage(self, wrld, exit_pos, monsters):
		if exit_pos is None:
			return False
		player_len = self._race_path_length(wrld, (self.x, self.y), exit_pos)
		if player_len is None:
			return False
		monster_best = None
		for m in monsters or []:
			mlen = self._race_path_length(wrld, m, exit_pos)
			if mlen is None:
				continue
			if monster_best is None or mlen < monster_best:
				monster_best = mlen
		# Advantage if no monster path, or strictly shorter than best monster path
		return monster_best is None or player_len < monster_best

	def _race_path_length(self, wrld, start, goal):
		# 8-directional BFS ignoring bombs/explosions; walls block
		if start == goal:
			return 0
		w, h = wrld.width(), wrld.height()
		visited = set([start])
		q = deque()
		q.append((start, 0))
		dirs = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
		while q:
			(pos, d) = q.popleft()
			x, y = pos
			for dx, dy in dirs:
				nx, ny = x + dx, y + dy
				if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
					if not wrld.wall_at(nx, ny):
						if (nx, ny) == goal:
							return d + 1
						visited.add((nx, ny))
						q.append(((nx, ny), d + 1))
		return None

	def find_exit(self, wrld):
		for x in range(wrld.width()):
			for y in range(wrld.height()):
				if wrld.exit_at(x, y):
					return (x, y)
		return None

	def find_monsters(self, wrld):
		positions = []
		for x in range(wrld.width()):
			for y in range(wrld.height()):
				if wrld.monsters_at(x, y):
					positions.append((x, y))
		return positions

	def chebyshev_distance(self, a, b):
		return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

	def _min_chebyshev_distance(self, pos, monsters):
		if not monsters:
			return float('inf')
		return min(self.chebyshev_distance(pos, m) for m in monsters)

	def _update_bomb_state(self, wrld):
		# Decrement post-blast cooldown if running
		if self.post_blast_cooldown > 0:
			self.post_blast_cooldown -= 1

		# If we have a recorded bomb position, see if bomb still exists there
		if self.owned_bomb_pos is not None:
			bx, by = self.owned_bomb_pos
			bomb_obj = wrld.bomb_at(bx, by)
			if bomb_obj is None and self.placed_bomb_this_flee:
				# Bomb at our recorded position disappeared -> detonation happened
				# Enforce 3-turn danger window for the explosion cross
				if self.post_blast_cooldown == 0:
					self.post_blast_cooldown = 3

	def _bomb_restrictions_active(self, wrld):
		# Restrictions are only considered while fleeing (as per spec)
		if not self.fleeing:
			return False
		if self.owned_bomb_pos is None:
			return self.post_blast_cooldown > 0
		bx, by = self.owned_bomb_pos
		bomb_obj = wrld.bomb_at(bx, by)
		if bomb_obj is not None and bomb_obj.timer <= 1:
			return True
		return self.post_blast_cooldown > 0

	def _bomb_effects_active(self, wrld):
		# Active from placement time until explosion clears (3 turns after detonation)
		if self.owned_bomb_pos is not None:
			bx, by = self.owned_bomb_pos
			bomb_obj = wrld.bomb_at(bx, by)
			# While the bomb exists at our recorded position, effects are considered active
			if bomb_obj is not None:
				return True
			# After it disappears, keep active during the post-blast cooldown window
			return self.post_blast_cooldown > 0
		# If we somehow have no recorded bomb, only the cooldown matters
		return self.post_blast_cooldown > 0

	def _any_bomb_exists(self, wrld):
		for x in range(wrld.width()):
			for y in range(wrld.height()):
				if wrld.bomb_at(x, y) is not None:
					return True
		return False

	def _cell_is_safe_while_fleeing(self, wrld, nx, ny):
		# Basic bounds/walkability/safety
		if not (0 <= nx < wrld.width() and 0 <= ny < wrld.height()):
			return False
		if wrld.wall_at(nx, ny):
			return False
		if wrld.explosion_at(nx, ny):
			return False
		if wrld.bomb_at(nx, ny):
			return False
		# Do not enter any imminent blast cross from any bomb
		if self._imminent_blast_at(nx, ny, wrld):
			return False

		# Additional fleeing-specific bomb restrictions
		if self._bomb_restrictions_active(wrld):
			# Active bomb about to explode OR within 3-turn post-blast window
			target_bomb_pos = self.owned_bomb_pos if self.owned_bomb_pos is not None else None
			if target_bomb_pos is None and self.post_blast_cooldown > 0:
				# We no longer know the exact bomb pos? Keep last known
				target_bomb_pos = self.owned_bomb_pos
			if target_bomb_pos is not None:
				bx, by = target_bomb_pos
				if nx == bx or ny == by:
					return False
		return True

	def _choose_flee_move(self, wrld, exit_pos, monsters):
		# Consider 8 neighbors and the option to wait in place if nothing else is safe
		candidates = []
		directions = [
			(0, -1), (1, -1), (1, 0), (1, 1),
			(0, 1), (-1, 1), (-1, 0), (-1, -1)
		]
		for dx, dy in directions:
			nx, ny = self.x + dx, self.y + dy
			if self._cell_is_safe_while_fleeing(wrld, nx, ny):
				candidates.append((nx, ny))
		# If no safe move, allow staying if current cell is safe
		if not candidates and self._cell_is_safe_while_fleeing(wrld, self.x, self.y):
			candidates.append((self.x, self.y))

		if not candidates:
			# No safe moves at all; fallback to staying in place (may be unsafe but nothing better)
			return (self.x, self.y)

		# Maximize distance to nearest monster (Chebyshev). Recompute per turn as monsters move.
		best_dist = -1
		best_cells = []
		for cell in candidates:
			dist = self._min_chebyshev_distance(cell, monsters) if monsters else float('inf')
			if dist > best_dist:
				best_dist = dist
				best_cells = [cell]
			elif dist == best_dist:
				best_cells.append(cell)

		if len(best_cells) == 1 or exit_pos is None:
			return best_cells[0]

		# Tie-breaker: shorter A* path to goal wins (without violating safety filters)
		best_cell = best_cells[0]
		best_len = self._path_length(self.astar_pathfinding(wrld, best_cell, exit_pos))
		for cell in best_cells[1:]:
			plen = self._path_length(self.astar_pathfinding(wrld, cell, exit_pos))
			if plen is not None and (best_len is None or plen < best_len):
				best_cell = cell
				best_len = plen
		return best_cell

	def _path_length(self, path):
		if not path:
			return None
		return len(path)

	def get_neighbors(self, pos, wrld):
		neighbors = []
		x, y = pos
		directions = [
			(0, -1), (1, -1), (1, 0), (1, 1),
			(0, 1), (-1, 1), (-1, 0), (-1, -1)
		]
		for dx, dy in directions:
			nx, ny = x + dx, y + dy
			if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
				if (not wrld.wall_at(nx, ny) and
					not wrld.bomb_at(nx, ny) and
					not wrld.explosion_at(nx, ny) and
					not self._imminent_blast_at(nx, ny, wrld)):
					neighbors.append((nx, ny))
		return neighbors

	def _imminent_blast_at(self, nx, ny, wrld):
		# Avoid stepping into any bomb row/column when bomb will detonate imminently (<=1)
		for bx in range(wrld.width()):
			for by in range(wrld.height()):
				bomb = wrld.bomb_at(bx, by)
				if bomb is not None and bomb.timer <= 1:
					if nx == bx or ny == by:
						return True
		return False

	def astar_pathfinding(self, wrld, start, goal):
		# A* with 8-directional movement; diagonal cost = 1 (1 turn), heuristic = Chebyshev
		open_heap = [(0, start)]
		heapq.heapify(open_heap)
		came_from = {}
		g_score = {start: 0}
		f_score = {start: self.chebyshev_distance(start, goal)}
		in_open = {start}
		closed = set()

		while open_heap:
			_, current = heapq.heappop(open_heap)
			if current in in_open:
				in_open.remove(current)

			if current == goal:
				path = []
				while current in came_from:
					path.append(current)
					current = came_from[current]
				path.reverse()
				return path

			closed.add(current)
			for neighbor in self.get_neighbors(current, wrld):
				if neighbor in closed:
					continue
				# Step cost: 1 per move (including diagonal)
				tentative_g = g_score[current] + 1
				if neighbor not in g_score or tentative_g < g_score[neighbor]:
					came_from[neighbor] = current
					g_score[neighbor] = tentative_g
					f_score[neighbor] = tentative_g + self.chebyshev_distance(neighbor, goal)
					if neighbor not in in_open:
						heapq.heappush(open_heap, (f_score[neighbor], neighbor))
						in_open.add(neighbor)
		return []


