from typing import Counter
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from awpy.visualization.plot import plot_map, position_transform
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.quiver import Quiver
from matplotlib.text import Text
import matplotlib

from models.data_manager import DataManager
from models.position_tracker import PositionTracker
from models.routine import DEFAULT_ROUTINE_LENGTH, Routine
from models.routine_tracker import RoutineTracker, TilizedRoutine
from models.side_type import SideType

class VisualizationManager:
    dm: DataManager
    fig: Figure
    axes: Axes
    
    current_round_index: int
    current_frame_index: int

    lines: list[Line2D] # Tracks lines for precise removal
    path_collections: list[PathCollection] # Tracks scatter plot points for precise removal
    text: list[Text] # Tracks text for precise removal

    visualized_routine_length: int
    do_visualize_routines: bool

    _position_tracker: PositionTracker | None
    position_tracker_drawings: PathCollection | None

    _routine_tracker: RoutineTracker | None
    routine_tracker_line_drawings: list[Quiver]
    routine_tracker_tile_drawings: PathCollection | None

    def __init__(self, dm: DataManager, fig: Figure, axes: Axes, visualized_routine_length: int = DEFAULT_ROUTINE_LENGTH):
        self.dm = dm
        self.fig = fig
        self.axes = axes
        
        self.current_round_index = 0
        self.current_frame_index = 0

        if visualized_routine_length < 0:
            raise ValueError('Visualized routine length cannot be negative.')
        self.visualized_routine_length = visualized_routine_length
        self.do_visualize_routines = False

        self.lines = list()
        self.path_collections = list()
        self.text = list()
        
        self._position_tracker = None
        self.position_tracker_drawings = None

        self._routine_tracker = None
        self.routine_tracker_line_drawings = list()
        self.routine_tracker_tile_drawings = None
    
    @classmethod
    def from_data_manager(cls, dm: DataManager) -> 'VisualizationManager':
        """Create a VisualizationManager object from an instantiated DataManager object."""
        fig, axes = plot_map(
            map_name=dm.get_map_name(),
            map_type='simpleradar',
            dark=True
        )
        return cls(dm, fig, axes)

    def render(self):
        """Render the map. Primarily for use in CLI contexts."""
        self.fig.show()

    def draw_routine(self, routine: Routine, fmt: str = '', **kwargs) -> Axes:
        """Draws a routine on the map. `fmt` is a format string following matplotlib fmt string notation, and kwargs can be used to add additional format options (overwriting any conflicting options from the format string)."""

        map_name = self.dm.get_map_name()

        transformed_x = [position_transform(map_name, xpos, 'x') for xpos in routine.x]
        transformed_y = [position_transform(map_name, ypos, 'y') for ypos in routine.y]

        self.lines.extend(self.axes.plot(transformed_x, transformed_y, fmt, **kwargs))
        return self.axes
    
    def toggle_routine_visualization(self):
        """Toggles whether routines are visualized or not."""
        self.do_visualize_routines = not self.do_visualize_routines
    
    @property
    def position_tracker(self) -> PositionTracker | None:
        """The position tracker that the visualization manager is using to draw heatmaps."""
        return self._position_tracker
    
    @position_tracker.setter
    def position_tracker(self, position_tracker: PositionTracker):
        """Sets the position tracker for the visualization manager to use in heatmap drawing."""
        self._position_tracker = position_tracker
    
    def _clear_position_heatmap_drawings(self):
        """Clears the position tracker drawings from the figure. 
        This is separate from `_clear_all_drawings` because drawing and clearing heatmaps should be separate from drawing and clearing player positions."""
        if self.position_tracker_drawings is not None:
            self.position_tracker_drawings.remove()
            self.position_tracker_drawings = None

    def draw_position_heatmap(self, **kwargs) -> Axes:
        """Draws a heatmap of player positions on the map based on the data in `self._position_tracker`. `**kwargs` are passed to the `scatter` function."""
        if self._position_tracker is None:
            raise ValueError('Position tracker is not set.')
        
        # Clear any existing heatmap drawings
        self._clear_position_heatmap_drawings()

        # Adding 0.5 to the tile coordinates to counteract a phenomena in which tiles are drawn with a small offset towards the top left corner of the map
        transformed_x = [(tile[0] + 0.5) * self._position_tracker.tile_length for tile in self._position_tracker.tile_activity_counter.keys()]
        transformed_y = [(tile[1] + 0.5) * self._position_tracker.tile_length for tile in self._position_tracker.tile_activity_counter.keys()]
        
        # Make the point color go from black to red based on the number of times the tile was visited
        # To use a colormap, we need a list of values between 0 and 1. Matplotlib uses the colormap to map these values to colors.
        # We want tiles with more visits to be "hotter" - for most colormaps brighter colors are produced by values closer to 1.
        # To do this, we scale the visit counts using the maximum visit count to produce values within that range.
        maximum_visit_count = max(self._position_tracker.tile_activity_counter.values() or [1]) # If there are no visits, set the maximum visit count to 1 to avoid division by zero
        scaled_visit_values = [count/maximum_visit_count for count in self._position_tracker.tile_activity_counter.values()]

        self.position_tracker_drawings = self.axes.scatter(transformed_x, transformed_y, c=scaled_visit_values, marker=MarkerStyle('s', 'full'), s=self._position_tracker._tile_length, alpha=0.5, cmap='YlOrRd', **kwargs)
        return self.axes
    
    @property
    def routine_tracker(self) -> RoutineTracker | None:
        """The routine tracker that the visualization manager is using to draw heatmaps."""
        return self._routine_tracker
    
    @routine_tracker.setter
    def routine_tracker(self, routine_tracker: RoutineTracker):
        """Sets the routine tracker for the visualization manager to use in heatmap drawing."""
        self._routine_tracker = routine_tracker
    
    def _clear_routine_heatmap_drawings(self):
        """Clears the routine tracker drawings from the figure. 
        This is separate from `_clear_all_drawings` because drawing and clearing heatmaps should be separate from drawing and clearing player positions."""
        if self.routine_tracker_tile_drawings is not None:
            self.routine_tracker_tile_drawings.remove()
            self.routine_tracker_tile_drawings = None

        for line in self.routine_tracker_line_drawings:
            line.remove()
        self.routine_tracker_line_drawings.clear()
    
    def draw_routine_tile_heatmap(self, **kwargs) -> Axes:
        """Draws a heatmap of player routines originating from each alive player on the map based on the data in `self._routine_tracker`. `**kwargs` are passed to the `scatter` function."""
        if self._routine_tracker is None:
            raise ValueError('Routine tracker is not set.')

        # Clear any existing heatmap drawings
        self._clear_routine_heatmap_drawings()

        alive_player_tiles: set[tuple[int, int]] = set()

        player_info_lists = self.dm.get_player_info_lists(self.current_round_index, self.current_frame_index)
        for player in (player_info_lists[SideType.T] + player_info_lists[SideType.CT]):
            if player['isAlive'] is False:
                continue
            
            tile_x = int(position_transform(self.dm.get_map_name(), player['x'], 'x') / self._routine_tracker.tile_length)
            tile_y = int(position_transform(self.dm.get_map_name(), player['y'], 'y') / self._routine_tracker.tile_length)

            alive_player_tiles.add((tile_x, tile_y))

            routines_originating_from_player_tile = self._routine_tracker.tile_routine_counter[(tile_x, tile_y)]
            print(f'{player["name"]} has {len(routines_originating_from_player_tile)} routines originating from tile ({tile_x}, {tile_y}).')

        activity_surrounding_alive_player_tiles: Counter[tuple[int, int]] = Counter()
        for tile in alive_player_tiles:
            for routine in self._routine_tracker.tile_routine_counter[tile]:
                for tile in zip(routine.tilized_x, routine.tilized_y):
                    activity_surrounding_alive_player_tiles[tile] += 1
        
        transformed_x = [(tile[0] + 0.5) * self._routine_tracker.tile_length for tile in activity_surrounding_alive_player_tiles.keys()]
        transformed_y = [(tile[1] + 0.5) * self._routine_tracker.tile_length for tile in activity_surrounding_alive_player_tiles.keys()]

        most_common_routine_count = max(activity_surrounding_alive_player_tiles.values() or [1]) # If there are no routines, set the most common routine count to 1 to avoid division by zero
        scaled_visit_values = [count/most_common_routine_count for count in activity_surrounding_alive_player_tiles.values()]

        self.routine_tracker_tile_drawings = self.axes.scatter(transformed_x, transformed_y, c=scaled_visit_values, marker=MarkerStyle('s', 'full'), s=self._routine_tracker.tile_length, alpha=0.75, cmap='YlOrRd', **kwargs)
        return self.axes
    
    def draw_routine_line_heatmap(self, **kwargs) -> Axes:
        """Draws a heatmap of player routines originating from each alive player on the map based on the data in `self._routine_tracker`. `**kwargs` are passed to the `plot` function."""
        if self._routine_tracker is None:
            raise ValueError('Routine tracker is not set.')
        
        # Clear any existing heatmap drawings
        self._clear_routine_heatmap_drawings()

        alive_player_tiles: set[tuple[int, int]] = set()

        player_info_lists = self.dm.get_player_info_lists(self.current_round_index, self.current_frame_index)
        for player in (player_info_lists[SideType.T] + player_info_lists[SideType.CT]):
            if player['isAlive'] is False:
                continue
            
            tile_x = int(position_transform(self.dm.get_map_name(), player['x'], 'x') / self._routine_tracker.tile_length)
            tile_y = int(position_transform(self.dm.get_map_name(), player['y'], 'y') / self._routine_tracker.tile_length)

            alive_player_tiles.add((tile_x, tile_y))

            routines_originating_from_player_tile = self._routine_tracker.tile_routine_counter[(tile_x, tile_y)]
            print(f'From {player["name"]}\'s tile, ({tile_x}, {tile_y}), {len(routines_originating_from_player_tile)} routines start.')

        routines_from_alive_player_tiles: Counter[TilizedRoutine] = Counter()
        for tile in alive_player_tiles:
            routines_from_alive_player_tiles += self._routine_tracker.tile_routine_counter[tile]
        
        most_common_routine_count = max(routines_from_alive_player_tiles.values() or [1]) # If there are no routines, set the most common routine count to 1 to avoid division by zero
        print(f'The most common routine count is {most_common_routine_count}.')
        
        # Pylance doesn't recognize the colormaps attribute of matplotlib, so I'm (begrudgingly) using a type ignore here.
        colormap = matplotlib.colormaps['YlOrRd'] # type: ignore
        
        for routine, count in routines_from_alive_player_tiles.items():
            transformed_x = [(tile[0] + 0.5) * self._routine_tracker.tile_length for tile in zip(routine.tilized_x, routine.tilized_y)]
            transformed_y = [(tile[1] + 0.5) * self._routine_tracker.tile_length for tile in zip(routine.tilized_x, routine.tilized_y)]
            scaled_color_value = count/most_common_routine_count
            color = colormap(scaled_color_value)
            print(f'Routine with count {count} has color {color} ({scaled_color_value}).')
            # Draw arrows so we can see the direction of the routine
            self.routine_tracker_line_drawings.append(
                self.axes.quiver(
                    transformed_x[:-1], transformed_y[:-1], [transformed_x[i+1] - transformed_x[i] for i in range(len(transformed_x) - 1)], [transformed_y[i+1] - transformed_y[i] for i in range(len(transformed_y) - 1)],
                    color=color, width=0.0025, **kwargs
                )
            )
        
        return self.axes
    
    def clear_heatmap_drawings(self):
        """Clears all heatmap-related drawings from the figure."""
        self._clear_position_heatmap_drawings()
        self._clear_routine_heatmap_drawings()
    
    def _clear_frame_related_drawings(self):
        """Clears all frame-related drawings (e.g. player positions, grenades - i.e. non-heatmap related drawings) from the figure."""
        for line in self.lines:
            line.remove()
        self.lines.clear()
        
        for collection in self.path_collections:
            collection.remove()
        self.path_collections.clear()

        for text in self.text:
            text.remove()
        self.text.clear()

    def _draw_frame(self) -> Axes:
        """Draws the current frame. Raises a ValueError if the current round index or the current frame index is out of bounds."""
        max_rounds = self.dm.get_round_count()
        max_frames = self.dm.get_frame_count(self.current_round_index)
        if self.current_round_index >= max_rounds:
            raise ValueError(f'Round index {self.current_round_index} is out of bounds. Max rounds: {max_rounds}')
        if self.current_frame_index >= max_frames:
            raise ValueError(f'Frame index {self.current_frame_index} is out of bounds. Max frames: {max_frames}')

        self._clear_frame_related_drawings()

        map_name = self.dm.get_map_name()

        # I wanted a function that for x = 0 returned 1, decreased linearly for a while, then asymptotically approached 0.
        # I wanted this behavior because it means extremely long routines won't be too cluttered as the oldest frames will be almost invisible, 
        # but I also wanted a clear, steady decrease of opacity for the most recent frames in the routine.
        alpha_function = lambda x: max(1 - 0.1*x, 1/(x + 1))

        routine_length = self.visualized_routine_length if self.do_visualize_routines else 0
        for frame_index_subtrahend in range(0, routine_length + 1):
            # Ensure we don't try to access a frame index that doesn't exist
            if self.current_frame_index - frame_index_subtrahend < 0:
                break

            frame_index = self.current_frame_index - frame_index_subtrahend

            player_info_lists = self.dm.get_player_info_lists(self.current_round_index, frame_index)
            t_side_players = player_info_lists[SideType.T]
            ct_side_players = player_info_lists[SideType.CT]

            transformed_t_x = [position_transform(map_name, player['x'], 'x') for player in t_side_players]
            transformed_t_y = [position_transform(map_name, player['y'], 'y') for player in t_side_players]
            self.path_collections.append(self.axes.scatter(transformed_t_x, transformed_t_y, c='goldenrod', alpha=alpha_function(frame_index_subtrahend)))

            transformed_ct_x = [position_transform(map_name, player['x'], 'x') for player in ct_side_players]
            transformed_ct_y = [position_transform(map_name, player['y'], 'y') for player in ct_side_players]
            self.path_collections.append(self.axes.scatter(transformed_ct_x, transformed_ct_y, c='lightblue', alpha=alpha_function(frame_index_subtrahend)))

            # Draw player names only for the most recent frame
            if frame_index_subtrahend == 0:
                for index, player in enumerate(t_side_players):
                    self.text.append(self.axes.text(transformed_t_x[index], transformed_t_y[index], player['name'], fontsize=10, ha='center', va='bottom', color='white'))
                for index, player in enumerate(ct_side_players):
                    self.text.append(self.axes.text(transformed_ct_x[index], transformed_ct_y[index], player['name'], fontsize=10, ha='center', va='bottom', color='white'))

        bomb_info = self.dm.get_bomb_info(self.current_round_index, self.current_frame_index)
        bomb_x = position_transform(map_name, bomb_info['x'], 'x')
        bomb_y = position_transform(map_name, bomb_info['y'], 'y')
        self.path_collections.append(self.axes.scatter(bomb_x, bomb_y, c='red')) # Maybe pick a better color for the bomb as fire grenades are also red

        # Plot grenades
        current_frame_tick = self.dm.get_frame(self.current_round_index, self.current_frame_index)['tick']

        grenade_color_map = {
            'Incendiary Grenade': 'red',
            'Molotov': 'red',
            'Smoke Grenade': 'gray',
            'HE Grenade': 'green',
            'Flashbang': 'gold',
        }

        thrower_color_map = {
            SideType.T: 'goldenrod',
            SideType.CT: 'lightblue',
        }

        for grenade in self.dm.get_grenade_events(self.current_round_index):
            if grenade['throwTick'] <= current_frame_tick <= grenade['destroyTick']:
                start_x = position_transform(map_name, grenade['throwerX'], 'x')
                start_y = position_transform(map_name, grenade['throwerY'], 'y')
                end_x = position_transform(map_name, grenade['grenadeX'], 'x')
                end_y = position_transform(map_name, grenade['grenadeY'], 'y')
                grenade_color = grenade_color_map[grenade['grenadeType']]
                thrower_color = thrower_color_map[SideType.from_str(grenade['throwerSide'])]
                self.lines.extend(self.axes.plot([start_x, end_x], [start_y, end_y], color=grenade_color))
                self.path_collections.append(self.axes.scatter(end_x, end_y, color=grenade_color, edgecolors=thrower_color))

        return self.axes
    
    def draw_round_start(self, round_index: int) -> Axes:
        """Draws the positions of players at the start of the given round number."""
        self.current_round_index = round_index
        self.current_frame_index = 0
        return self._draw_frame()
    
    def revisualize(self):
        """Revisualizes the current round and frame."""
        return self._draw_frame()

    def progress_visualization(self) -> bool:
        """Progresses the visualization by one frame.
        Returns True if the visualization is still playing, False if it has reached the end."""
        max_rounds = self.dm.get_round_count()
        max_frames = self.dm.get_frame_count(self.current_round_index)
        self.current_frame_index += 1
        # If we're past the last frame of the round, go to the next round
        if self.current_frame_index >= max_frames:
            self.current_frame_index = 0
            self.current_round_index += 1
        # Can't progress if we've progressed past the last round
        if self.current_round_index >= max_rounds:
            # To make sure our state values are never out of bounds, set the current round index to the last round and the current frame index to the last frame of the last round
            self.current_round_index = max_rounds - 1
            final_round_frame_count = self.dm.get_frame_count(self.current_round_index)
            self.current_frame_index = final_round_frame_count - 1
            return False
        self._draw_frame()
        return True
        