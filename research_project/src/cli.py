from typing import Literal
from models.data_manager import EXAMPLE_DEMO_PATH, DataManager
from models.position_tracker import PositionTracker
from models.routine import DEFAULT_ROUTINE_LENGTH
from models.routine_tracker import RoutineTracker, TilizedRoutine
from models.visualization_manager import VisualizationManager
from awpy.visualization.plot import position_transform

def test_routine_drawing():
    """Test drawing a routine from the example demo file."""
    dm = DataManager(EXAMPLE_DEMO_PATH, do_validate=False)
    vizm = VisualizationManager.from_data_manager(dm)

    team_routines = dm.get_all_team_routines(16, DEFAULT_ROUTINE_LENGTH)
    t_side_player_one_routines = team_routines.t_side.routines[0]
    for index, routine in enumerate(t_side_player_one_routines):
        print(f'x_{index}: {",".join(map(str, routine.x))}')
        print(f'y_{index}: {",".join(map(str, routine.y))}')

    t_side_player_one_first_routine = t_side_player_one_routines[0] 
    vizm.draw_routine(t_side_player_one_first_routine)
    vizm.render()
    input('Press Enter to go back to the menu.')

def test_heatmap_generation():
    """Test generating and visualizing a heatmap of player positions throughout the game from the example demo file."""
    data_manager = DataManager(EXAMPLE_DEMO_PATH, do_validate=False)
    tracker = PositionTracker(data_manager.get_map_name(), 20)
    for round_index in range(data_manager.get_round_count()):
        for frame_index in range(data_manager.get_frame_count(round_index)):
            for player_list in data_manager.get_player_info_lists(round_index, frame_index).values():
                for player_info in player_list:
                    transformed_x, transformed_y = position_transform(tracker.map_name, player_info['x'], 'x'), position_transform(tracker.map_name, player_info['y'], 'y')
                    tracker.add_transformed_coordinates(transformed_x, transformed_y)
    vizm = VisualizationManager.from_data_manager(data_manager)
    vizm.position_tracker = tracker
    vizm.draw_position_heatmap()
    vizm.render()
    input('Press Enter to go back to the menu.')

def test_routine_heatmap(do_aggregate_multiple_files: bool = False, heatmap_type: Literal['tiles'] | Literal['lines'] = 'tiles', file_aggregation_limit: int | None = 20):
    """Test generating and visualizing a heatmap of player routines throughout the game from the example demo file.
    Can optionally aggregate routines from multiple demo files in a directory.
    The heatmap can be drawn with either tiles or lines."""
    data_manager = DataManager(EXAMPLE_DEMO_PATH, do_validate=False)
    tile_length = 20
    if do_aggregate_multiple_files:
        tracker = RoutineTracker.aggregate_routines_from_directory(EXAMPLE_DEMO_PATH.parent / 'lan', data_manager.get_map_name(), tile_length, routine_length=DEFAULT_ROUTINE_LENGTH, limit=file_aggregation_limit)
    else:
        tracker = RoutineTracker(data_manager.get_map_name(), tile_length)
    for round_index in range(data_manager.get_round_count()):
        team_routines = data_manager.get_all_team_routines(round_index, DEFAULT_ROUTINE_LENGTH)
        for team in (team_routines.t_side, team_routines.ct_side):
            for player_routines in team.routines:
                for routine in player_routines:
                    tracker.add_routine(TilizedRoutine(routine, tile_length))
    vizm = VisualizationManager.from_data_manager(data_manager)
    # Progressing the visualization to some arbitrary point in the game so the heatmap is more interesting than what it would look like from spawn positions.
    vizm.draw_round_start(0)
    for _ in range(25):
        vizm.progress_visualization()
    vizm.routine_tracker = tracker
    if heatmap_type == 'tiles':
        vizm.draw_routine_tile_heatmap()
    else:
        vizm.draw_routine_line_heatmap()
    vizm.render()
    input('Press Enter to go back to the menu.')

if __name__ == '__main__':
    print('Hello.')

    while True:
        selection = input("""Select an option:
        1. Test drawing a routine from the example demo file.
        2. Test generating and visualizing a heatmap of player positions.
        3. Test generating and visualizing a heatmap of player routines.
        Q. Quit the program.
              """)
        match selection:
            case '1':
                test_routine_drawing()
            case '2':
                test_heatmap_generation()
            case '3':
                do_aggregate_multiple_files = False
                heatmap_type = 'tiles'
                file_aggregation_limit = None

                answer = input('Do you want to aggregate routines from multiple demo files? (y/N)').lower()
                match answer:
                    case 'y':
                        do_aggregate_multiple_files = True
                        try:
                            file_aggregation_limit = int(input('Enter the number of files to aggregate routines from:'))
                        except ValueError:
                            print('Invalid input. Defaulting to 20 files.')
                            file_aggregation_limit = 20
                    case _:
                        print('Only using routines from the example demo file.')
                
                answer = input('Do you want to draw the heatmap with tiles (t) or lines (l)? (t/l)').lower()
                match answer:
                    case 't':
                        heatmap_type = 'tiles'
                    case 'l':
                        heatmap_type = 'lines'
                    case _:
                        print('Invalid input. Defaulting to tiles.')
                
                test_routine_heatmap(do_aggregate_multiple_files, heatmap_type, file_aggregation_limit)
            case 'Q':
                break
            case _:
                print('Invalid selection. Please try again.')
        
    print('Goodbye.')
