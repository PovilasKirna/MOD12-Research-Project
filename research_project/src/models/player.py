from dataclasses import dataclass
from awpy.types import PlayerInfo

@dataclass
class Player:
    alive: bool
    hp: int
    active_weapon: str # TODO: Maybe enum this
    equipment_value: int

    @classmethod
    def from_player_info(cls, player_info: PlayerInfo) -> 'Player':
        return cls(
            alive=player_info['isAlive'],
            hp=player_info['hp'],
            active_weapon=player_info['activeWeapon'],
            equipment_value=player_info['equipmentValue']
        )
