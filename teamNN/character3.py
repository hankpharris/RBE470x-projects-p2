import sys
import heapq
import math
from collections import deque

#This is the start of our code, algorithmically controlling behavior
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

class Character3(CharacterEntity):

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.path = []
        self.path_index = 0
        self.fleeing = False
        self.flee_target = None
        self.just_placed_bomb = False

    def do(self, wrld):
        # Find the exit
        exit_pos = self.find_exit(wrld)
        if not exit_pos:
            return  # No exit found, stay still
        
        # If we're at the exit, we're done
        if (self.x, self.y) == exit_pos:
            return
        
        # Check if we're in danger from bombs/explosions
        if self.is_in_danger(wrld):
            if not self.fleeing:
                print("ENTERING FLEE MODE - BOMB DETECTED!")
            self.fleeing = True
            self.flee_target = self.find_safest_reachable_position(wrld, (self.x, self.y))
        
        # Check if we should place a bomb (before movement)
        if self.should_place_bomb(wrld):
            print("PLACING BOMB!")
            self.place_bomb()
            print("ENTERING FLEE MODE - BOMB PLACED!")
            self.fleeing = True
            self.flee_target = self.find_safest_reachable_position(wrld, (self.x, self.y))
            self.just_placed_bomb = True
            # Immediately flee this tick to avoid exiting prematurely
            self.flee_behavior(wrld)
            self.just_placed_bomb = False
            return
        
        # If we're fleeing, prioritize safety
        if self.fleeing:
            if self.flee_target and not self.is_in_danger(wrld) and not self.just_placed_bomb:
                # We've reached safety, stop fleeing
                print("EXITING FLEE MODE - SAFE!")
                self.fleeing = False
                self.flee_target = None
            else:
                # Continue fleeing to safety
                print("FLEEING TO SAFETY...")
                self.flee_behavior(wrld)
                # Clear the one-tick bomb flag if it was set earlier
                self.just_placed_bomb = False
                return
        
        # Normal pathfinding behavior (only if not fleeing)
        self.path = self.astar_pathfinding(wrld, (self.x, self.y), exit_pos)
        self.path_index = 0
        
        if self.path:
            next_pos = self.path[0]
            dx = next_pos[0] - self.x
            dy = next_pos[1] - self.y
            self.move(dx, dy)
            self.path_index = 1
        else:
            # No path found: we've likely reached a local goal minimum (closest reachable to exit)
            # Place a bomb to clear obstacles, then immediately enter flee mode
            print("LOCAL MINIMUM REACHED - PLACING BOMB AND FLEEING!")
            self.place_bomb()
            self.fleeing = True
            self.flee_target = self.find_safest_reachable_position(wrld, (self.x, self.y))
            self.just_placed_bomb = True
            # Execute flee behavior in the same tick for responsiveness
            self.flee_behavior(wrld)
            self.just_placed_bomb = False
            return

    def find_exit(self, wrld):
        """Find the exit position in the world"""
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.exit_at(x, y):
                    return (x, y)
        return None

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def should_place_bomb(self, wrld):
        """Check if we should place a bomb (when 5 cells from a monster)"""
        # Check if there's already a bomb on the map
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.bomb_at(x, y):
                    return False  # Don't place bomb if one already exists
        
        # Debug: Print monster positions
        print(f"Character at ({self.x}, {self.y}) checking for monsters:")
        
        # Check if there's a monster at exactly 5 cells away
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                monsters = wrld.monsters_at(x, y)
                if monsters:
                    distance = self.manhattan_distance((self.x, self.y), (x, y))
                    print(f"  Monster at ({x}, {y}), distance: {distance}")
                    if distance <= 5:  # Exactly 5 cells away
                        print("  BOMB SHOULD BE PLACED AT 5 CELLS!")
                        return True
        
        return False

    def is_in_danger(self, wrld):
        """Check if we're in danger - safety condition is whether a bomb exists"""
        # Check current position for explosion
        if wrld.explosion_at(self.x, self.y):
            return True
        
        # Check if any bomb exists in the world
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.bomb_at(x, y):
                    return True
        
        return False

    def find_safe_position(self, wrld):
        """Find a safe position away from bombs and explosions"""
        best_pos = None
        best_distance = 0
        
        # Look for positions that are far from bombs and explosions
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if (not wrld.wall_at(x, y) and 
                    not wrld.bomb_at(x, y) and 
                    not wrld.explosion_at(x, y)):
                    
                    # Calculate minimum distance to any bomb
                    min_bomb_distance = float('inf')
                    for bx in range(wrld.width()):
                        for by in range(wrld.height()):
                            bomb = wrld.bomb_at(bx, by)
                            if bomb:
                                bomb_dist = self.manhattan_distance((x, y), (bx, by))
                                min_bomb_distance = min(min_bomb_distance, bomb_dist)
                    
                    # If this position is safer than our current best
                    if min_bomb_distance > best_distance:
                        best_distance = min_bomb_distance
                        best_pos = (x, y)
        
        print(f"Safe position found: {best_pos} (distance from bombs: {best_distance})")
        return best_pos

    def find_safest_reachable_position(self, wrld, start):
        """Find the safest reachable position using flee danger weights (reachable-aware)."""
        # Precompute bombs and monsters for tie-breakers
        bomb_positions = []
        for bx in range(wrld.width()):
            for by in range(wrld.height()):
                if wrld.bomb_at(bx, by):
                    bomb_positions.append((bx, by))
        monster_positions = []
        for mx in range(wrld.width()):
            for my in range(wrld.height()):
                if wrld.monsters_at(mx, my):
                    monster_positions.append((mx, my))

        def min_manhattan_distance(pos, positions):
            if not positions:
                return float('inf')
            px, py = pos
            return min(abs(px - qx) + abs(py - qy) for (qx, qy) in positions)

        visited = set([start])
        queue = deque([start])

        # Initialize with current position's danger
        initial_bomb_cost = self.get_bomb_danger_cost(start, wrld) * 2
        initial_monster_cost = self.get_monster_proximity_cost(start, wrld) * 0.5
        best_pos = start
        best_danger = initial_bomb_cost + initial_monster_cost
        best_bomb_dist = min_manhattan_distance(start, bomb_positions)
        best_mon_dist = min_manhattan_distance(start, monster_positions)

        while queue:
            current = queue.popleft()

            # Evaluate safety at this node
            node_bomb_cost = self.get_bomb_danger_cost(current, wrld) * 2
            node_monster_cost = self.get_monster_proximity_cost(current, wrld) * 0.5
            node_danger = node_bomb_cost + node_monster_cost

            # Small nudge to avoid choosing current tile on bomb-placement tick
            adjusted_danger = node_danger + (0.1 if (self.just_placed_bomb and current == start) else 0.0)

            node_bomb_dist = min_manhattan_distance(current, bomb_positions)
            node_mon_dist = min_manhattan_distance(current, monster_positions)

            if (
                adjusted_danger < best_danger or
                (
                    adjusted_danger == best_danger and (
                        node_bomb_dist > best_bomb_dist or
                        (node_bomb_dist == best_bomb_dist and node_mon_dist > best_mon_dist)
                    )
                )
            ):
                best_danger = adjusted_danger
                best_pos = current
                best_bomb_dist = node_bomb_dist
                best_mon_dist = node_mon_dist

            # Explore neighbors (8-directional reachability)
            for neighbor in self.get_neighbors(current, wrld):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        return best_pos

    def flee_behavior(self, wrld):
        """Flee to safety when in danger"""
        # Check if we're adjacent to the exit and can reach it
        exit_pos = self.find_exit(wrld)
        if exit_pos:
            exit_distance = self.manhattan_distance((self.x, self.y), exit_pos)
            if exit_distance == 1:  # Adjacent to exit
                print("FLEE MODE: Adjacent to exit, moving to exit!")
                dx = exit_pos[0] - self.x
                dy = exit_pos[1] - self.y
                self.move(dx, dy)
                return
            elif exit_distance <= 3:  # Close to exit, prioritize reaching it
                print(f"FLEE MODE: Close to exit (distance: {exit_distance}), prioritizing exit!")
                # Use normal A* to reach exit quickly
                self.path = self.astar_pathfinding(wrld, (self.x, self.y), exit_pos)
                self.path_index = 0
                if self.path:
                    next_pos = self.path[0]
                    dx = next_pos[0] - self.x
                    dy = next_pos[1] - self.y
                    print(f"Fleeing to exit: moving from ({self.x}, {self.y}) to ({next_pos[0]}, {next_pos[1]})")
                    self.move(dx, dy)
                    self.path_index = 1
                return
        
        # Re-evaluate safest reachable flee target every tick (monsters/bombs move)
        self.flee_target = self.find_safest_reachable_position(wrld, (self.x, self.y))
        if not self.flee_target:
            print("No flee target, using normal pathfinding")
            return

        print(f"Fleeing to target: {self.flee_target} (recomputed)")
        # Use A* to find path to safety, with 
        # safest-reachable fallback
        self.path = self.astar_flee_pathfinding(wrld, (self.x, self.y), self.flee_target)
        self.path_index = 0
        
        if self.path:
            next_pos = self.path[0]
            dx = next_pos[0] - self.x
            dy = next_pos[1] - self.y
            print(f"Fleeing: moving from ({self.x}, {self.y}) to ({next_pos[0]}, {next_pos[1]})")
            self.move(dx, dy)
            self.path_index = 1
        else:
            # Recompute a reachable-safe target and try once more
            alt_target = self.find_safest_reachable_position(wrld, (self.x, self.y))
            if alt_target and alt_target != self.flee_target:
                print(f"Recomputing flee target to reachable: {alt_target}")
                self.flee_target = alt_target
                self.path = self.astar_flee_pathfinding(wrld, (self.x, self.y), self.flee_target)
                self.path_index = 0
                if self.path:
                    next_pos = self.path[0]
                    dx = next_pos[0] - self.x
                    dy = next_pos[1] - self.y
                    print(f"Fleeing: moving from ({self.x}, {self.y}) to ({next_pos[0]}, {next_pos[1]})")
                    self.move(dx, dy)
                    self.path_index = 1
                    return
            print("No flee path found!")

    def get_monster_proximity_cost(self, pos, wrld):
        """Calculate additional cost for being near monsters"""
        cost = 0
        monster_penalty = 50  # Very high penalty for being near monsters
        
        # Check in a 2-cell radius around the position
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                check_x, check_y = pos[0] + dx, pos[1] + dy
                if (0 <= check_x < wrld.width() and 
                    0 <= check_y < wrld.height()):
                    monsters = wrld.monsters_at(check_x, check_y)
                    if monsters:  # Check if list is not empty
                        # Higher penalty for closer monsters
                        distance = max(abs(dx), abs(dy))
                        if distance == 1:
                            cost += monster_penalty * 2  # Adjacent cells
                        else:
                            cost += monster_penalty  # 2-cell radius
        return cost

    def get_bomb_danger_cost(self, pos, wrld):
        """Calculate cost for being near bombs"""
        cost = 0
        bomb_penalty = 1000  # Extremely high penalty for being near bombs
        
        # Check all bombs in the world
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                bomb = wrld.bomb_at(x, y)
                if bomb:
                    distance = self.manhattan_distance(pos, (x, y))
                    if distance <= wrld.expl_range:
                        # Higher penalty for being closer to bomb explosion range
                        if distance <= 1:
                            cost += bomb_penalty * 10  # Directly adjacent
                        elif distance <= 2:
                            cost += bomb_penalty * 5   # Very close
                        else:
                            cost += bomb_penalty       # Within explosion range
        
        return cost

    def get_neighbors(self, pos, wrld):
        """Get valid neighboring positions (8-directional movement)"""
        neighbors = []
        x, y = pos
        
        # 8-directional movement: N, NE, E, SE, S, SW, W, NW
        directions = [
            (0, -1),   # North
            (1, -1),   # Northeast
            (1, 0),    # East
            (1, 1),    # Southeast
            (0, 1),    # South
            (-1, 1),   # Southwest
            (-1, 0),   # West
            (-1, -1)   # Northwest
        ]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            # Check bounds
            if (0 <= new_x < wrld.width() and 0 <= new_y < wrld.height()):
                # Check if position is walkable (not a wall, not occupied by bomb/explosion)
                if (not wrld.wall_at(new_x, new_y) and 
                    not wrld.bomb_at(new_x, new_y) and 
                    not wrld.explosion_at(new_x, new_y)):
                    neighbors.append((new_x, new_y))
        
        return neighbors

    def astar_pathfinding(self, wrld, start, goal):
        """A* pathfinding algorithm with 8-directional movement and monster avoidance"""
        # Priority queue: (f_score, position)
        open_set = [(0, start)]
        heapq.heapify(open_set)
        
        # Dictionaries to store costs and paths
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, goal)}
        came_from = {}
        
        # Set to track positions in open set
        open_set_positions = {start}
        
        # Set to track explored positions
        closed_set = set()
        
        # Track the closest reachable position to the goal encountered during search
        best_node_towards_goal = start
        best_heuristic_to_goal = self.manhattan_distance(start, goal)

        while open_set:
            # Get position with lowest f_score
            current_f, current = heapq.heappop(open_set)
            
            if current in open_set_positions:
                open_set_positions.remove(current)
            
            # If we reached the goal, reconstruct path
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            closed_set.add(current)
            
            # Check all neighbors
            for neighbor in self.get_neighbors(current, wrld):
                if neighbor in closed_set:
                    continue
                
                # Calculate movement cost (diagonal movement costs more)
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                if dx == 1 and dy == 1:  # Diagonal movement
                    move_cost = math.sqrt(2)
                else:  # Orthogonal movement
                    move_cost = 1
                
                # Add monster proximity cost
                monster_cost = self.get_monster_proximity_cost(neighbor, wrld)
                tentative_g = g_score[current] + move_cost + monster_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    neighbor_to_goal_h = self.manhattan_distance(neighbor, goal)
                    f_score[neighbor] = tentative_g + neighbor_to_goal_h

                    # Update closest reachable node toward goal
                    if neighbor_to_goal_h < best_heuristic_to_goal:
                        best_heuristic_to_goal = neighbor_to_goal_h
                        best_node_towards_goal = neighbor
                    
                    # Add to open set if not already there
                    if neighbor not in open_set_positions:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_positions.add(neighbor)
        
        # No exact path found: return path to closest reachable point toward the goal (if any)
        if best_node_towards_goal != start and best_node_towards_goal in came_from:
            current = best_node_towards_goal
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        return []

    def astar_flee_pathfinding(self, wrld, start, goal):
        """A* pathfinding algorithm optimized for fleeing to safety"""
        # Priority queue: (f_score, position)
        open_set = [(0, start)]
        heapq.heapify(open_set)
        
        # Dictionaries to store costs and paths
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, goal)}
        came_from = {}
        
        # Set to track positions in open set
        open_set_positions = {start}
        
        # Set to track explored positions
        closed_set = set()
        
        # Track the safest reachable position encountered during search
        # Safety is evaluated using the same flee-weighted danger metrics
        safest_node = start
        safest_cost = float('inf')
        # Secondary tie-breaker: heuristic distance to the flee goal (prefer closer when equally safe)
        safest_node_heuristic = self.manhattan_distance(start, goal)

        while open_set:
            # Get position with lowest f_score
            current_f, current = heapq.heappop(open_set)
            
            if current in open_set_positions:
                open_set_positions.remove(current)
            
            # If we reached the goal, reconstruct path
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            closed_set.add(current)
            
            # Check all neighbors
            for neighbor in self.get_neighbors(current, wrld):
                if neighbor in closed_set:
                    continue
                
                # Calculate movement cost (diagonal movement is cheaper in flee mode)
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                if dx == 1 and dy == 1:  # Diagonal movement
                    move_cost = 0.8  # Cheaper diagonal movement for fleeing
                else:  # Orthogonal movement
                    move_cost = 1.0
                
                # For fleeing, prioritize bomb avoidance over monster avoidance
                bomb_cost = self.get_bomb_danger_cost(neighbor, wrld) * 2  # Double bomb penalty
                monster_cost = self.get_monster_proximity_cost(neighbor, wrld) * 0.5  # Reduce monster penalty
                tentative_g = g_score[current] + move_cost + bomb_cost + monster_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    neighbor_to_goal_h = self.manhattan_distance(neighbor, goal)
                    f_score[neighbor] = tentative_g + neighbor_to_goal_h

                    # Update safest reachable node according to flee danger metrics
                    neighbor_safety_cost = bomb_cost + monster_cost
                    if (neighbor_safety_cost < safest_cost or
                        (neighbor_safety_cost == safest_cost and neighbor_to_goal_h < safest_node_heuristic)):
                        safest_cost = neighbor_safety_cost
                        safest_node = neighbor
                        safest_node_heuristic = neighbor_to_goal_h
                    
                    # Add to open set if not already there
                    if neighbor not in open_set_positions:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_positions.add(neighbor)
        
        # No exact path found: return path to the safest reachable position (if any)
        if safest_node != start and safest_node in came_from:
            current = safest_node
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        return []


