from dataclasses import dataclass
from models.side_type import SideType
from models.player import Player

@dataclass
class RoundStats:
    players: list[Player]
    winning_side: SideType
    round_end_reason: str # TODO: Maybe enum this
    opponents_alive: int
    opponent_equipment_value: int
    clock_time: str
