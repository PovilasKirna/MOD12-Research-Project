import logging

from numpy import std

from metrics.base_metric import BaseMetric
from models.data_manager import DataManager

logger = logging.getLogger(__name__)

class VelocityDeviationMetric(BaseMetric):
  """
  VelocityDeviationMetric provides a metric to estimate the standard deviation of velocity among players.
  """
  def process_metric_frame(self,dm: DataManager,
                           round_idx: int,
                           frame_idx: int,
                           plot_metric: bool = False) -> float:
    """
    Args:
      dm: DataManager instance.
      round_idx: The round index.
      frame_idx:  The frame index.
      plot_metric: Plot the frame?

    Returns: The Velocity Deviation among all players in T-side

    """
    logger.debug("Calculating %s metrics for round %d, frame %d" % (self.__class__.__name__, round_idx, frame_idx))
    frame = dm.get_frame(round_idx, frame_idx)
    players = frame["t"]["players"]
    velocities = [abs(player["velocityX"])+abs(player["velocityX"]) for player in players]
    return std(velocities)
