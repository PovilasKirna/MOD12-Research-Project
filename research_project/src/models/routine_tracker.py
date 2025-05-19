from collections import defaultdict, Counter
from pathlib import Path
from typing import overload
from models.data_manager import DataManager, get_map_name_from_demo_file_without_parsing
from models.demo_metadata import DemoMetadata
from models.routine import DEFAULT_ROUTINE_LENGTH, FrameCount, Routine
from awpy.visualization.plot import position_transform

class TilizedRoutine(Routine):
    """An extension of the Routine class that includes the tilized x and y values for the routine - that is, the x and y values transformed into tile coordinates."""
    _tile_length: int
    _tilized_x: list[int] # The x values of the routine transformed into tile coordinates.
    _tilized_y: list[int] # The y values of the routine transformed into tile coordinates.

    def __init__(self, routine: Routine, tile_length: int):
        super().__init__(routine.player_name, routine.team, routine.map_name, list(zip(routine.x, routine.y)))
        self._tile_length = tile_length
        # Transforming coordinates now as bucketing them into tiles and then transforming tile coordinates sounds like it would be less accurate - not sure if this feeling is true, though.
        self._tilized_x = [int(position_transform(routine.map_name, x, 'x') / tile_length) for x in routine.x]
        self._tilized_y = [int(position_transform(routine.map_name, y, 'y') / tile_length) for y in routine.y]

    @property
    def tile_length(self) -> int:
        """The length of each tile. As each tile is a square, this value is used for both the width and height of each tile."""
        return self._tile_length
    
    @property
    def tilized_x(self) -> list[int]:
        """Returns a list of x values for the routine transformed into tile coordinates."""
        return self._tilized_x
    
    @property
    def tilized_y(self) -> list[int]:
        """Returns a list of y values for the routine transformed into tile coordinates."""
        return self._tilized_y
    
    @overload
    def __getitem__(self, index: int) -> tuple[int, int]:
        """Returns the tilized x and y values at the given index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[int, int]]:
        """Returns a list of tilized x and y value tuples at the given slice."""
        ...

    def __getitem__(self, index: int | slice) -> tuple[int, int] | list[tuple[int, int]]:
        """Determines which indexing method to use based on the value of the index parameter and returns the correct amount of x and y tuple values"""
        if isinstance(index, int):
            return self._tilized_x[index], self._tilized_y[index]
        elif isinstance(index, slice):
            return list(zip(self._tilized_x[index.start:index.stop:index.step], self._tilized_y[index.start:index.stop:index.step]))
        else:
            raise TypeError("Index must be an integer or slice.")
    
    # I want TilizedRoutines that have the same sequence of tilized x and y values to be considered equal, for set operations.
    def __hash__(self) -> int:
        return hash((tuple(self._tilized_x), tuple(self._tilized_y)))
    
    def __eq__(self, other: 'TilizedRoutine') -> bool:
        return self._tilized_x == other.tilized_x and self._tilized_y == other.tilized_y


class RoutineTracker:
    _map_name: str
    _tile_length: int
    _routine_length: int
    _tile_routine_counter: defaultdict[tuple[int, int], Counter[TilizedRoutine]] # A mapping from tile coordinates to a Counter object for tracking the number of times a routine starting from that tile has been counted.
    _metadata: list[DemoMetadata] # Metadata for the demos that the routines were extracted from.

    def __init__(self, map_name: str, tile_length: int, routine_length: int = DEFAULT_ROUTINE_LENGTH):
        self._map_name = map_name
        self._tile_length = tile_length
        self._routine_length = routine_length
        self._tile_routine_counter = defaultdict(Counter)
        self._metadata = list()
    
    @classmethod
    def from_data_manager(cls, dm: DataManager, tile_length: int, routine_length: FrameCount = DEFAULT_ROUTINE_LENGTH) -> 'RoutineTracker':
        """Instantiates a RoutineTracker object from a DataManager object, a tile length, and an optional routine length, adding all the routines in the game to the tracker."""
        tracker = cls(dm.get_map_name(), tile_length, routine_length)
        for round_index in range(dm.get_round_count()):
            team_routines = dm.get_all_team_routines(round_index, routine_length)
            for team in (team_routines.t_side, team_routines.ct_side):
                for player_routines in team.routines:
                    for routine in player_routines:
                        tracker.add_routine(TilizedRoutine(routine, tile_length))
        tracker._metadata = [DemoMetadata.from_data_manager(dm)]
        return tracker

    @classmethod
    def aggregate_routines_from_directory(cls, directory_path: Path, map_name: str, tile_length: int, routine_length: FrameCount = DEFAULT_ROUTINE_LENGTH, limit: int | None = None) -> 'RoutineTracker':
        """Aggregates all the routines from a directory of demo files into a single RoutineTracker object.
        If a limit is provided, only the first limit number of files will be processed."""
        tracker = RoutineTracker(map_name, tile_length, routine_length)

        files_processed = 0
        total_file_count = len(list(directory_path.iterdir()))
        demos_aggregated = 0
        total_demos_to_aggregate = min(limit, total_file_count) if limit is not None else total_file_count
        
        for file_path in directory_path.iterdir():
            if file_path.suffix == '.json':
                # Skip demos that aren't for the map we're interested in.
                if get_map_name_from_demo_file_without_parsing(file_path) != map_name:
                    files_processed += 1
                    continue

                try:
                    dm = DataManager(file_path, do_validate=False)
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
                    files_processed += 1
                    continue

                tracker += RoutineTracker.from_data_manager(dm, tile_length, routine_length)
                files_processed += 1
                demos_aggregated += 1
                print(f"Processed {file_path.name} - {files_processed}/{total_file_count} files processed, {demos_aggregated} demos aggregated.")
                if total_demos_to_aggregate is not None and demos_aggregated >= total_demos_to_aggregate:
                    break

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
    def routine_length(self) -> int:
        """The length of each routine that is being tracked."""
        return self._routine_length
    
    @property
    def tile_routine_counter(self) -> dict[tuple[int, int], Counter[TilizedRoutine]]:
        """The dictionary mapping tile coordinates to Counter objects that keep track of how many times each routine was taken from that tile."""
        return self._tile_routine_counter
    
    @property
    def metadata(self) -> list[DemoMetadata]:
        """The metadata for the demos that the routines were extracted from."""
        return self._metadata
    
    def add_routine(self, routine: TilizedRoutine) -> int:
        """Increments the counter for the tile that the given routine's starting position falls into. 
        The position values stored in the routine object are assumed to be transformed into the map's coordinate system via the position_transform function from the awpy module.
        Returns the new count."""

        tile_x, tile_y = routine[0]
        self._tile_routine_counter[(tile_x, tile_y)][routine] += 1
        return self._tile_routine_counter[(tile_x, tile_y)][routine]
    
    def __len__(self) -> int:
        """Returns the total number of routines tracked by the RoutineTracker."""
        return sum(sum(counter.values()) for counter in self._tile_routine_counter.values())
    
    def __add__(self, other: 'RoutineTracker') -> 'RoutineTracker':
        """Combines two RoutineTracker objects by adding their tile_routine_counter attributes together and combining the metadata lists."""
        if self._map_name != other.map_name or self._tile_length != other.tile_length or self._routine_length != other.routine_length:
            raise ValueError("RoutineTrackers must be for the same map, have identical tile lengths, and have identical routine lengths to be combined.")
        combined_tracker = RoutineTracker(self._map_name, self._tile_length, self._routine_length)
        for tile, counter in self._tile_routine_counter.items():
            combined_tracker._tile_routine_counter[tile] += counter
        for tile, counter in other.tile_routine_counter.items():
            combined_tracker._tile_routine_counter[tile] += counter
        combined_tracker._metadata = self._metadata + other._metadata
        return combined_tracker
