from dataclasses import dataclass
from typing import NewType, overload

from models.side_type import SideType

# This NewType exists to make sure we don't pass any kind of number in for FrameCount.
# When creating a value of type FrameCount, we must be expressly aware that this variable we're making represents a frame count.
FrameCount = NewType('FrameCount', int)

DEFAULT_ROUTINE_LENGTH = FrameCount(5)

@dataclass(eq=False)
class Routine:
    """A class representing a player's trajectory - that is, multiple points."""
    player_name: str
    team: SideType
    map_name: str
    _x: list[float]
    _y: list[float]
    
    def __init__(self, player_name: str, team: SideType, map_name: str, positions: list[tuple[float, float]] | None = None):
        self.player_name = player_name
        self.team = team
        self.map_name = map_name
        if positions is None:
            self._x = []
            self._y = []
        else:
            self._x, self._y = zip(*positions)
            
    @property
    def x(self) -> list[float]:
        """Returns a list of x values for the routine."""
        return self._x
    
    @property
    def y(self) -> list[float]:
        """Returns a list of y values for the routine."""
        return self._y

    @overload
    def __getitem__(self, index: int) -> tuple[float, float]:
        """Returns the x and y values at the given index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[float, float]]:
        """Returns a list of x and y value tuples at the given slice."""
        ...

    def __getitem__(self, index: int | slice) -> tuple[float, float] | list[tuple[float, float]]:
        """Determines which indexing method to use based on the value of the index parameter and returns the correct amount of x and y tuple values"""
        if isinstance(index, int):
            return self._x[index], self._y[index]
        elif isinstance(index, slice):
            return list(zip(self._x[index.start:index.stop:index.step], self._y[index.start:index.stop:index.step]))
        else:
            raise TypeError("Index must be an integer or slice.")
    
    def __len__(self) -> int:
        """Returns the length of the routine."""
        # Because x and y values are ensured to be the same length in the constructor, we can just return the length of one of them.
        return len(self._x)
