from awpy.analytics.nav import generate_area_distance_matrix
from awpy.data import NAV
from awpy.visualization.plot import plot_map, position_transform
from matplotlib import pyplot as plt, patches



def _util_generate_area_distance_matrix(map_name: str) -> None:
  """This is a convenient method to generate an area distance matrix.

  **INFO**
  Make sure to save and copy the resulting file to your awpy python package distribution:
  ./python3.11/site-packages/awpy/data/nav

  Method can take a long time. See documentation for:
  generate_area_distance_matrix(map_name: str, *, save: bool = False) -> AreaMatrix

  Args:
    map_name: The map name.

  """
  generate_area_distance_matrix(map_name, save=True)



def _util_find_bomb_area_ids(map_name: str):
  f, ax = plot_map(map_name=map_name, map_type='simpleradar', dark=True)

  f, ax = plt.subplots(1, figsize=[64.0,48.0], subplot_kw={"facecolor": "black"})
  ax.set_xlim(0, 6400)
  ax.set_ylim(0, 4800)

  for a in NAV[map_name]:
    area = NAV[map_name][a]
    southEastX = position_transform(map_name, area["southEastX"], "x")
    northWestX = position_transform(map_name, area["northWestX"], "x")
    southEastY = position_transform(map_name, area["southEastY"], "y")
    northWestY = position_transform(map_name, area["northWestY"], "y")
    width = (southEastX - northWestX)
    height = (northWestY - southEastY)
    col = "None"
    if area["areaName"].startswith("Bombsite"):
      col = "red"
    rect = patches.Rectangle((northWestX, southEastY), width, height, linewidth=1, edgecolor="yellow",
                             facecolor=col)
    ax.add_patch(rect)
    if area["areaName"].startswith("Bombsite"):
      ax.text(northWestX, northWestY, str(a), fontsize=4, color="white")
  f.show()


