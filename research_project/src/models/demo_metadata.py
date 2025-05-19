from dataclasses import dataclass
from pathlib import Path

from models.data_manager import DataManager
from models.side_type import SideType

@dataclass(frozen=True)
class TeamPerformanceRecord:
    """Some basic information about how a team from a demo performed in the match."""
    name: str
    ending_score: int
    starting_side: SideType

@dataclass(frozen=True)
class DemoMetadata:
    path: Path
    match_id: str | None # Not every demo has a match ID, unfortunately.
    map_name: str
    round_count: int
    ct_team_info: TeamPerformanceRecord # Information for the team that started the map on the CT side.
    t_team_info: TeamPerformanceRecord # Same but for T side
    # TODO: Add more metadata fields as needed.

    @classmethod
    def from_data_manager(cls, dm: DataManager) -> 'DemoMetadata':
        """Extracts the demo metadata from the demo the DataManager is wrapping and returns a DemoMetadata object."""

        round_count = dm.get_round_count()
        team_names = dm.get_team_names(0)
        final_round_team_scores = dm.get_team_scores(round_count - 1)

        ct_info = TeamPerformanceRecord(
            name=team_names.ct_team_name,
            ending_score=final_round_team_scores.ct_score,
            starting_side=SideType.CT
        )
        t_info = TeamPerformanceRecord(
            name=team_names.t_team_name,
            ending_score=final_round_team_scores.t_score,
            starting_side=SideType.T
        )
        
        return cls(
            path=dm.file_path,
            match_id=dm.get_match_id(),
            map_name=dm.get_map_name(),
            round_count=round_count,
            ct_team_info=ct_info,
            t_team_info=t_info
        )

    def get_fields_for_table(self) -> dict[str, str | int]:
        """Returns a dictionary of fields that can be used to populate a table with the metadata."""
        return {
            'File Name': self.path.name,
            # Excluding map because all demos in a directory should be for the same map.
            'Round Count': self.round_count,
            'Team 1 (CT Start)': self.ct_team_info.name,
            'Team 1 Final Score': self.ct_team_info.ending_score,
            'Team 2 (T Start)': self.t_team_info.name,
            'Team 2 Final Score': self.t_team_info.ending_score
        }