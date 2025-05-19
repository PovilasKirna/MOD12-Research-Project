import logging
import os

from awpy.data import NAV
from typing_extensions import override

from metrics.base_metric import BaseMetric
from models.data_manager import DataManager

LOGGING_LEVEL = os.environ.get("LOGGING_INFO")
if LOGGING_LEVEL == "INFO":
  logging.basicConfig(level=logging.INFO)
elif LOGGING_LEVEL == "DEBUG":
  logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


## TODO: Add total distance not only to previous frame;
## TODO: Todo, estimate metric depending on CT vs. T side.

class DistanceMetric(BaseMetric):

  def __init__(self, cumulative: bool = True):
    self.cumulative = cumulative
    self.total_distance = 0.0
    self.previous_positions = None

  @override
  def process_metric_frame(self, dm: DataManager, round_idx: int, frame_idx: int, plot_metric: bool = False) -> float:
    """ Calculates the metric for a single frames in a round.

    **WARNING**: This method only works if frame are iterated by correct order !!

    Args:
      dm: DataManager that hosts all game data.
      round_idx: The round index.
      frame_idx: The frame index.
      plot_metric: True = plot the round chart; False = no plot

    Returns: A single metric value.

    """
    logger.debug("Calculating %s metrics for round %d, frame %d" % (self.__class__.__name__, round_idx, frame_idx))
    map_name = dm.get_map_name()

    if map_name not in NAV:
      raise ValueError("Map not found.")

    frames = dm._get_frames(round_idx)
    if frame_idx >= len(frames):
      raise ValueError(f"Frame index {frame_idx} out of bounds (max index is {len(frames) - 1})")

    # first frame is always 0.0 by definition (there cannot be a preceding frame to get a distance delta)
    if frame_idx < 1 or self.previous_positions == None:
      # save previous positions for next frame and return 0.0.
      self.previous_positions = {
        player["name"]: (player["x"], player["y"], player["z"])
        for player in frames[frame_idx]["t"]["players"]}
      return 0.0

    current_frame = frames[frame_idx]
    previous_frame = frames[frame_idx-1]

    # players are indexed at different positions each frame, so we need to remember them by name...
    # tuples are (x,y,z)
    current_positions = {
       player["name"]: (player["x"], player["y"], player["z"])
    for player in current_frame["t"]["players"] }

    delta_positions = {
      player_name: (
          current_positions[player_name][0] - self.previous_positions[player_name][0],
          current_positions[player_name][1] - self.previous_positions[player_name][1],
          current_positions[player_name][2] - self.previous_positions[player_name][2]
      ) for player_name in current_positions }

    # we use absolute values
    metric = sum([
      sum((abs(delta[0]), abs(delta[1]), abs(delta[2])))
      for delta in delta_positions.values()
      ])

    # save total distance and previous positions for next frame and return metric value
    self.previous_positions = current_positions
    self.total_distance += metric

    if plot_metric:
      # TODO: Think of plotting, e.g. a simple scatterplot / line with delta distance
      pass

    if self.cumulative:
      return self.total_distance
    else:
        return metric



  def process_metric_round(self, dm: DataManager, round_idx: int, plot_metric: bool = False) -> list[float]:
    """Calculates the metric for all frames in a round. Base implementation just aggregates all single frame
    values into a list.

    Args:
      dm: DataManager that hosts all game data.
      round_idx: The round index.
      plot_metric: True = plot the round chart; False = no plot

    Returns: List of metric values.
    """
    # We have to reset total distance before calling the base implementation
    self.total_distance = 0.0
    return super().process_metric_round(dm, round_idx, plot_metric)




if __name__ == "__main__":
  logger.setLevel(logging.INFO)
  from pathlib import Path
  import os
  # demo_path = os.path.join(os.getcwd(), 'demos/esta/lan/de_dust2/00e7fec9-cee0-430f-80f4-6b50443ceacd.json')
  # demo_path = Path(__file__).parent / '../demos/esta/lan/de_dust2/00e7fec9-cee0-430f-80f4-6b50443ceacd.json'
  demo_path = Path(__file__).parent / '../../demos/esta/lan/de_dust2/00e7fec9-cee0-430f-80f4-6b50443ceacd.json'
  dm = DataManager(demo_path, do_validate=False)
  # testframe = dm.get_frame(5, 8)
  # map_control_fig, map_control_axes = plot_frame_map_control(dm.get_map_name(), testframe, plot_type='players')
  # map_control_fig.show()

  tdm = DistanceMetric(cumulative=True)
  tdm.process_metric_round(dm, 4, True)
