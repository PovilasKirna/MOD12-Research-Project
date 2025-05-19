from dataclasses import dataclass, field
from awpy.types import KillAction, DamageAction, GrenadeAction, BombAction, WeaponFireAction, FlashAction

@dataclass
class RoundActions:
    kills: list[KillAction] = field(default_factory=list)
    damages: list[DamageAction] = field(default_factory=list)
    grenades: list[GrenadeAction] = field(default_factory=list)
    bomb_events: list[BombAction] = field(default_factory=list)
    weapon_fires: list[WeaponFireAction] = field(default_factory=list)
    flashes: list[FlashAction] = field(default_factory=list)
