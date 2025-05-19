from dataclasses import dataclass
from models.routine import Routine

@dataclass
class TeamRoutines:
    """A class tracking per-player routine lists for one team in a match."""
    routines: list[list[Routine]]

    @classmethod
    def from_routines_list(cls, routines: list[list[Routine]]) -> 'TeamRoutines':
        """Instantiates object from a list of routines. The list should be a list of lists, where each inner list contains the routines for a player."""
        return cls(routines)

    def get_player_routines(self, player_index: int) -> list[Routine]:
        """Returns the routines for the given player index. Raises a ValueError if the player index is invalid."""
        if player_index < 0 or player_index >= len(self.routines):
            raise ValueError(f'Invalid player index: {player_index}. Must be between 0 and {len(self.routines) - 1}.')
        return self.routines[player_index]

@dataclass
class BothTeamsRoutines:
    """A small, wrapper class for holding routine-tracking objects for both teams in a match."""
    t_side: TeamRoutines
    ct_side: TeamRoutines
