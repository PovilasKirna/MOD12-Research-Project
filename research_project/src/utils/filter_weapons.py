import json
from pathlib import Path

known_weapons = {
    "": 0,
    "Decoy Grenade": 1,
    "AK-47": 2,
    "M4A1": 3,
    "Incendiary Grenade": 4,
    "Knife": 5,
    "MAC-10": 6,
    "USP-S": 7,
    "Tec-9": 8,
    "AWP": 9,
    "Glock-18": 10,
    "SSG 08": 11,
    "HE Grenade": 12,
    "Galil AR": 13,
    "C4": 14,
    "Smoke Grenade": 15,
    "Molotov": 16,
    "P250": 17,
    "Flashbang": 18,
    "SG 553": 19,
    "Desert Eagle": 20,
    "Zeus x27": 21,
    "CZ75 Auto": 22,
    "M4A4": 23,
    "Five-SeveN": 24,
    "AUG": 25,
    "FAMAS": 26,
    "MP9": 27,
    "G3SG1": 28,
    "UMP-45": 29,
    "MP5-SD": 30,
    "Dual Berettas": 31,
    "P2000": 32,
    "MP7": 33,
    "Nova": 34,
    "XM1014": 35,
    "MAG-7": 36,
    "Sawed-Off": 37,
    "SCAR-20": 38,
    "PP-Bizon": 39,
    "M249": 40,
    "Negev": 41,
    "Taser": 42,
    "R8 Revolver": 43,
    "M4A1-S": 44,
}

demo_dir = Path("research_project/demos/dust2")
unknown_weapons = {}

for file in demo_dir.glob("*.json"):
    with open(file, "r") as f:
        try:
            demo = json.load(f)
        except Exception as e:
            print(f"⚠️ Skipped {file.name}: {e}")
            continue

    for round in demo.get("gameRounds", []):
        for frame in round.get("frames", []):
            t_team = frame.get("t")
            players = t_team.get("players") if isinstance(t_team, dict) else None
            if not isinstance(players, list):
                continue
            for player in players:
                weapon = player.get("activeWeapon", "")
                if weapon not in known_weapons:
                    unknown_weapons.setdefault(file.name, set()).add(weapon)

print("\n📋 Unknown weapons found:")
for fname, weapons in unknown_weapons.items():
    print(f"{fname}: {', '.join(weapons)}")
