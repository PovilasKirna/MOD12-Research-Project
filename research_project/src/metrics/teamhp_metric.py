import logging

from metrics.base_metric import BaseMetric
from models.data_manager import DataManager

logger = logging.getLogger(__name__)

class TeamHpMetric(BaseMetric):
  """
  TeamHp provides a metric to estimate the total health points of a team.
  """
  def __init__(self, team: str):
    if team != "ct" and team != "t":
      raise ValueError("Team should be 't' or 'ct'.")
    self.teamside = team
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
    players = frame[self.teamside]["players"]
    hps = [player["hp"] for player in players]
    return sum(hps)
