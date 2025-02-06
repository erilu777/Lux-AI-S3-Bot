import numpy as np
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple

@dataclass
class MapInfo:
    """Track map features"""
    explored_tiles: Set[Tuple[int, int]] = None
    nebula_tiles: Set[Tuple[int, int]] = None
    energy_nodes: List[Tuple[int, int]] = None
    relic_nodes: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        self.explored_tiles = set()
        self.nebula_tiles = set()
        self.energy_nodes = []
        self.relic_nodes = []

class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.env_cfg = env_cfg
        self.map_info = MapInfo()
        
        # Game parameters we need to learn
        self.learned_params = {
            'unit_move_cost': None,
            'nebula_vision_reduction': None,
            'unit_sap_cost': None
        }
    
    def update_map_knowledge(self, obs):
        """Update what we know about the map"""
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_mask = np.array(obs["units_mask"][self.team_id])
        
        # Update explored areas
        for unit_id in np.where(unit_mask)[0]:
            pos = tuple(map(int, unit_positions[unit_id]))
            self.map_info.explored_tiles.add(pos)
            
        # Track relic nodes
        relic_mask = np.array(obs["relic_nodes_mask"])
        relic_positions = np.array(obs["relic_nodes"])
        for idx in np.where(relic_mask)[0]:
            pos = tuple(map(int, relic_positions[idx]))
            if pos not in self.map_info.relic_nodes:
                self.map_info.relic_nodes.append(pos)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        
        # Update knowledge
        self.update_map_knowledge(obs)
        
        # Get available units
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energy = np.array(obs["units"]["energy"][self.team_id])
        available_units = np.where(unit_mask)[0]

        # Simple exploration for now
        for unit_id in available_units:
            if unit_energy[unit_id] < 10:  # Ensure enough energy to move
                continue
                
            # TODO: 
            # - Implement more sophisticated exploration
            # - Implement unit actions
            # - Implement unit movement
            # - Implement targeting of relic nodes
            # - Iplement unit energy management
            # - Implement unit vision management
            # - Implement unit sap management
            # - Implement unit pathfinding
            # - Implement updating of learned parameters
            # - Implement more sophisticated world representation
            # - Implement more sophisticated unit representation
            # - Etc etc etc...

            # Move up     
            actions[unit_id] = [1, 0, 0]

        return actions