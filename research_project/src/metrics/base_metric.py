import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from awpy.data import NAV

from models.data_manager import DataManager

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
  """
  This class provides interface for all metrics. Each metric is calculated on the frame-level and then can be
  aggregated towards the round level. Base implementation of the round aggregation is just putting the frame
  level values in a list.
  """
  @abstractmethod
  def process_metric_frame(self, dm: DataManager, round_idx: int, frame_idx: int, plot_metric: bool = False) -> float:
    """Calculates the metric for a single frames in a round.

    Args:
      dm: DataManager that hosts all game data.
      round_idx: The round index.
      frame_idx: The frame index.
      plot_metric: True = plot the round chart; False = no plot

    Returns: A single metric value.
    """
    pass

  def process_metric_round(self, dm: DataManager, round_idx: int, plot_metric: bool = False) -> list[float]:
    """Calculates the metric for all frames in a round. Base implementation just aggregates all single frame
    values into a list.

    Args:
      dm: DataManager that hosts all game data.
      round_idx: The round index.
      plot_metric: True = plot the round chart; False = no plot

    Returns: List of metric values.
    """
    logger.info("Calculating %s metrics for round %d." % (self.__class__.__name__, round_idx))
    map_name = dm.get_map_name()

    if map_name not in NAV:
      raise ValueError("Map not found.")

    metric_values = []
    for frame_idx, frame in enumerate(dm.get_game_round(round_idx)["frames"] or []):
      try:
        metric = self.process_metric_frame(dm, round_idx, frame_idx)
        metric_values.append(metric)
      except ValueError as err:
        logger.warning(err)
        logger.warning("Ignoring frame %d and adding NA instead." % frame_idx)
        metric_values.append(None)

    if plot_metric:
      logger.info("Plotting %s metrics for round %d." % (self.__class__.__name__, round_idx))
      plt.scatter(np.arange(len(metric_values)), metric_values)
      plt.show()

    return metric_values
