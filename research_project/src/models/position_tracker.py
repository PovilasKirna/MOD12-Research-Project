from collections import Counter

from models.data_manager import DataManager

from awpy.visualization.plot import position_transform

class PositionTracker:
    """A class for tracking the cumulative amount of times players enter each tile on the map, with a configurable tile size."""
    _map_name: str
    _tile_length: int
    _tile_activity_counter: Counter[tuple[int, int]]

    def __init__(self, map_name: str, tile_length: int):
        self._map_name = map_name
        self._tile_length = tile_length
        self._tile_activity_counter = Counter()
    
    @classmethod
    def from_data_manager(cls, dm: DataManager, tile_length: int) -> 'PositionTracker':
        """Instantiates a PositionTracker object from a DataManager object and a tile length, adding the player positions from every game frame to the tracker."""
        tracker = cls(dm.get_map_name(), tile_length)
        for round_index in range(dm.get_round_count()):
            for frame_index in range(dm.get_frame_count(round_index)):
                for player_list in dm.get_player_info_lists(round_index, frame_index).values():
                    for player_info in player_list:
                        transformed_x, transformed_y = position_transform(tracker.map_name, player_info['x'], 'x'), position_transform(tracker.map_name, player_info['y'], 'y')
                        tracker.add_transformed_coordinates(transformed_x, transformed_y)
        return tracker
    
    @property
    def map_name(self) -> str:
        """The name of the map for which data is being tracked. 
        Useful for ensuring that the correct map is being used in visualization or analysis."""
        return self._map_name
    
    @property
    def tile_length(self) -> int:
        """The length of each tile. As each tile is a square, this value is used for both the width and height of each tile."""
        return self._tile_length
    
    @property
    def tile_activity_counter(self) -> Counter[tuple[int, int]]:
        """The Counter object that keeps track of how many times each tile has been visited. 
        The keys are tuples of the form (x, y) where x and y are the coordinates of the tile."""
        return self._tile_activity_counter

    def add_transformed_coordinates(self, x: float, y: float) -> int:
        """Increments the counter for the tile that the given player position coordinates fall into. 
        Assumes that the given coordinates are already transformed to the correct map's coordinate system via the position_transform function from the awpy module.
        Returns the new count."""
        tile_x = int(x / self._tile_length)
        tile_y = int(y / self._tile_length)
        self._tile_activity_counter[(tile_x, tile_y)] += 1
        return self._tile_activity_counter[(tile_x, tile_y)]
