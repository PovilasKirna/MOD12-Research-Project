import logging
import os

from awpy.analytics.nav import area_distance, find_closest_area
from awpy.data import NAV, AREA_DIST_MATRIX
from awpy.types import BombInfo
from awpy.visualization.plot import plot_map, position_transform
from matplotlib import patches
from typing_extensions import override

from metrics.base_metric import BaseMetric
from models.data_manager import DataManager

LOGGING_LEVEL = os.environ.get("LOGGING_INFO")
if LOGGING_LEVEL == "INFO":
  logging.basicConfig(level=logging.INFO)
elif LOGGING_LEVEL == "DEBUG":
  logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


## TODO: Find only bombsite area tiles where you can plant
## TODO: Todo, estimate metric depending on CT vs. T side.


class BombDistanceMetric(BaseMetric):
  """
  The BombDistanceMetric provides a metric that estimates the distance from the bomb's position to the
  closest bombsite.
  """
  @override
  def process_metric_frame(self, dm: DataManager, round_idx: int, frame_idx: int, plot_metric: bool = False) -> float:
    logger.debug("Calculating %s metrics for round %d, frame %d" % (self.__class__.__name__, round_idx, frame_idx))
    map_name = dm.get_map_name()

    if map_name not in NAV:
      raise ValueError("Map not found.")

    # find shorted bombsite:
    closest_bombsite_dist: float = float("Inf")
    closest_bombsite_areaid: int = -1
    bombinfo: BombInfo = dm.get_bomb_info(round_idx, frame_idx)
    bomb_coords = [bombinfo[key] for key in bombinfo.keys()]
    area_bomb = find_closest_area(map_name, point=bomb_coords, flat=False)
    geodesic_path = None

    ## Todo: find bombsite *plantable* area  with minimum distance from bomb

    for area_id in NAV[map_name]:
      area = NAV[map_name][area_id]
      if area["areaName"].startswith("Bombsite"):
        # Use Area Distance Matrix if available, since it is faster
        # area distance matrix uses str as keys
        # *TODO* copy pasted this from the create_graphy.py check if it works here as well
        area_a_str = str(area_id)
        area_b_str = str(area_bomb["areaId"])
        if (map_name in AREA_DIST_MATRIX
            and area_a_str in AREA_DIST_MATRIX[map_name]
            and area_b_str in AREA_DIST_MATRIX[map_name][area_a_str]):
          current_bombsite_dist = AREA_DIST_MATRIX[map_name][area_a_str][area_b_str]["geodesic"]
        # Else: calculate all distances pairwise
        else:
          if LOGGING_LEVEL == "DEBUG" and len(AREA_DIST_MATRIX) > 0: # this happened once, not sure if debug overhead is needed
            logger.debug("Area matrix exists but does not contain areaid: %d" % area_id)
          geodesic_path = area_distance(map_name=map_name, area_a=area_id, area_b=area_bomb["areaId"], dist_type="geodesic")
          current_bombsite_dist = geodesic_path["distance"]
        # Set closest area_id
        if current_bombsite_dist < closest_bombsite_dist:
          closest_bombsite_areaid = area_id
          closest_bombsite_dist = current_bombsite_dist

    if closest_bombsite_areaid < 0:
      raise ValueError("Could not find closest bombsite distance with bomb area id: %d in frame %d." % (area_bomb["areaId"], frame_idx))

    if not geodesic_path:
      geodesic_path = area_distance(map_name=map_name, area_a=closest_bombsite_areaid, area_b=area_bomb["areaId"], dist_type="geodesic")

    if not plot_metric:
      return closest_bombsite_dist

    logger.info("Plotting %s metrics for round %d, frame %d." % (self.__class__.__name__, round_idx, frame_idx))
    fig, ax = plot_map(map_name=map_name, map_type='simpleradar', dark=True)

    for a in NAV[map_name]:
      area = NAV[map_name][a]
      southEastX = position_transform(map_name, area["southEastX"], "x")
      northWestX = position_transform(map_name, area["northWestX"], "x")
      southEastY = position_transform(map_name, area["southEastY"], "y")
      northWestY = position_transform(map_name, area["northWestY"], "y")
      width = (southEastX - northWestX)
      height = (northWestY - southEastY)

      color = "None"
      if a in geodesic_path["areas"]:
        color = "red"

      rect = patches.Rectangle((northWestX, southEastY), width, height, linewidth=1, edgecolor="yellow",
                               facecolor=color)
      ax.add_patch(rect)

    fig.show()
    return closest_bombsite_dist



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

  bdm = BombDistanceMetric()

  bdm.process_metric_round(dm, 5, True)


  # area id 8968 is isolated
  # bdm.process_metric_frame(dm, 5, 3, True)




