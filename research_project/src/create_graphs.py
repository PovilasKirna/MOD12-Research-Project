import asyncio
import functools
import json
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Process
from pathlib import Path
from typing import Any

import stats
from awpy.analytics.nav import area_distance, find_closest_area
from awpy.data import AREA_DIST_MATRIX, NAV
from dotenv import load_dotenv
from models.data_manager import DataManager
from tqdm import tqdm
from utils.discord_webhook import send_progress_embed
from utils.download_demo_from_repo import get_demo_files_from_list
from utils.logging_config import get_logger

load_dotenv()

# Remove any default root handlers (they print to console)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set root level to WARNING or higher (to suppress DEBUG logs)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("discord").setLevel(logging.CRITICAL)
logging.getLogger("discord.webhook.async_").setLevel(logging.CRITICAL)

KEYS_ROUND_LEVEL = (
    "tFreezeTimeEndEqVal",
    "tRoundStartEqVal",
    "tRoundSpendMoney",
    "tBuyType",
)
KEYS_FRAME_LEVEL = ("tick", "seconds", "bombPlanted")
KEYS_PLAYER_LEVEL = (
    "x",
    "y",
    "z",
    "velocityX",
    "velocityY",
    "velocityZ",
    "viewX",
    "viewY",
    "hp",
    "armor",
    "activeWeapon",
    "totalUtility",
    "isAlive",
    "isDefusing",
    "isPlanting",
    "isReloading",
    "isInBombZone",
    "isInBuyZone",
    "equipmentValue",
    "equipmentValueFreezetimeEnd",
    "equipmentValueRoundStart",
    "cash",
    "cashSpendThisRound",
    "cashSpendTotal",
    "hasHelmet",
    "hasDefuse",
    "hasBomb",
)
KEYS_PER_NODE = KEYS_PLAYER_LEVEL + (
    "areaId",
    "nodeType",
)  # areaId and nodeType are added during processing

# DGL library requires int as node ids
# instead of randomly converting later, ensure that it is always 6,7,8, because
# players are 1-5
BOMB_NODE_INDEX = 6
BOMBSITE_A_NODE_INDEX = 7
BOMBSITE_B_NODE_INDEX = 8

# PyTorch can only handle numeric tensors
NODE_TYPE_PLAYER_INDEX = (
    1000  # players are more like bomb than targets (based on attributes)
)
NODE_TYPE_BOMB_INDEX = 900
NODE_TYPE_TARGET_INDEX = 1
WEAPON_ID_MAPPING = {  # TODO: add missing weapons
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
}


def process_round(
    dm: DataManager,
    round_idx: int,
    strategy_used: str = "unknown",
    queue=None,
    key=None,
    logger=None,
) -> list[list[Any]]:
    round = dm.get_game_round(round_idx)
    map_name = dm.get_map_name()

    # all variables on the round level --> graph data
    round_data = {key: round[key] for key in KEYS_ROUND_LEVEL}
    round_data["strategy_used"] = strategy_used

    frames = dm._get_frames(round_idx)

    # store crucial bomb events for later analysis and estimating correct round ingame seconds.
    bomb_event_data = stats.process_bomb_data(round)

    # iterate and process each frame
    graphs = []
    error_frame_count = 0
    total_frames = len(frames)
    # Remove local tqdm bar, progress will be reported via queue
    for frame_idx, frame in enumerate(frames):
        # check validity of frame
        valid_frame, err_text = stats.check_frame_validity(frame)
        if not valid_frame:
            logger.debug(
                "Skipping frame %d entirely because %s." % (frame_idx, err_text)
            )
            if queue and key:
                queue.put((key, 1))
            continue

        # all variables on the frame level are added to the graph level data.
        graph_data = {key: frame[key] for key in KEYS_FRAME_LEVEL} | round_data

        # include estimated seconds from bomb data for each frame
        if (
            bomb_event_data["bombTick"] != None
            and frame["tick"] >= bomb_event_data["bombTick"]
        ):
            graph_data["seconds"] = frame["seconds"] + bomb_event_data["bombSeconds"]
        else:
            graph_data["seconds"] = frame["seconds"]

        # all variables on the team and player level for the T side
        team = frame["t"]

        ### Create Node Data
        # iterate through all players, but keep them in same order every iteration
        nodes_data = {}
        edges_data = []
        for player_idx, player in enumerate(
            sorted(
                team["players"],
                key=lambda p: dm.get_player_idx_mapped(p["name"], "t", frame),
            )
        ):
            node_data = {key: player[key] for key in KEYS_PLAYER_LEVEL}
            node_data["areaId"] = find_closest_area(
                map_name, point=[node_data[key] for key in ("x", "y", "z")], flat=False
            )["areaId"]
            node_data["nodeType"] = NODE_TYPE_PLAYER_INDEX
            node_data["activeWeapon"] = map_weapon_to_id(node_data["activeWeapon"])
            nodes_data[player_idx] = node_data

        # add bomb node
        nodes_data[BOMB_NODE_INDEX] = dm.get_bomb_info(round_idx, frame_idx)
        nodes_data[BOMB_NODE_INDEX]["areaId"] = find_closest_area(
            map_name,
            point=[nodes_data[BOMB_NODE_INDEX][key] for key in ("x", "y", "z")],
            flat=False,
        )["areaId"]
        nodes_data[BOMB_NODE_INDEX]["nodeType"] = NODE_TYPE_BOMB_INDEX

        ### Create Edge Data
        # computes distances to bombsite
        try:
            distance_A, distance_B = distance_bombsites(dm, nodes_data, logger=logger)
        except ValueError as exc:
            logger.warning(
                "Frame %d (%f%%): %s" % (frame_idx, frame_idx / total_frames, exc)
            )
            logger.exception(f"Round {round_idx}, Frame {frame_idx}: {exc}")
            error_frame_count += 1
            if queue and key:
                queue.put((key, 1))
            continue  # skip errors
        for k in nodes_data.keys():
            edges_data.append((k, BOMBSITE_A_NODE_INDEX, {"dist": distance_A[k]}))
            edges_data.append((k, BOMBSITE_B_NODE_INDEX, {"dist": distance_B[k]}))

        # compute distances pairwise
        for node_a in reversed(nodes_data.keys()):
            for node_b in reversed(nodes_data.keys()):
                # ignore self loops
                if node_a == node_b:
                    continue
                edges_data.append(
                    (
                        node_a,
                        node_b,
                        {
                            "dist": _distance_internal(
                                map_name,
                                nodes_data[node_a]["areaId"],
                                nodes_data[node_b]["areaId"],
                                logger=logger,
                            )
                        },
                    )
                )

        # add target site nodes after all distance calcuations, so we can always just take the entire dict as input
        nodes_data[BOMBSITE_A_NODE_INDEX] = {"nodeType": NODE_TYPE_TARGET_INDEX}
        nodes_data[BOMBSITE_B_NODE_INDEX] = {"nodeType": NODE_TYPE_TARGET_INDEX}

        # fill up all keys with empty values, because all nodes need same attributes for DGL
        for k in nodes_data.keys():
            nodes_data[k] = fill_keys(nodes_data[k])  # merging dicts creates a copy

        # store data in convenient dict
        graph = {
            "graph_data": graph_data,
            "nodes_data": nodes_data,
            "edges_data": edges_data,
        }
        graphs.append(graph)
        if queue and key:
            queue.put((key, 1))
    # print("\n")
    return graphs


def map_weapon_to_id(weaponName: str):
    return WEAPON_ID_MAPPING[weaponName]


def fill_keys(target: dict):
    empty_dict = {key: 0 for key in KEYS_PER_NODE}  # None does not work as tensor
    return empty_dict | target  # right dict takes precedence


def distance_bombsites(dm: DataManager, nodes: dict, logger=None):
    logger.debug("Calculating shortest distances for %d nodes." % len(nodes))
    map_name = dm.get_map_name()

    if map_name not in NAV:
        raise ValueError("Map not found.")

    # find shortest distances to both bombsites:
    closest_distances_A = {key: float("Inf") for key in nodes}
    closest_distances_B = {key: float("Inf") for key in nodes}

    ## Todo: find bombsite *plantable* area  with minimum distance from bomb
    for map_area_id in NAV[map_name]:
        map_area = NAV[map_name][map_area_id]
        if map_area["areaName"].startswith("BombsiteA"):
            for key, value in nodes.items():
                target_area = nodes[key]["areaId"]
                current_bombsite_dist = _distance_internal(
                    map_name, map_area_id, target_area
                )
                # Set closest distance
                if current_bombsite_dist < closest_distances_A[key]:
                    closest_distances_A[key] = current_bombsite_dist

        elif map_area["areaName"].startswith("BombsiteB"):
            for key, value in nodes.items():
                target_area = nodes[key]["areaId"]
                current_bombsite_dist = _distance_internal(
                    map_name, map_area_id, target_area
                )
                # Set closest distance
                if current_bombsite_dist < closest_distances_B[key]:
                    closest_distances_B[key] = current_bombsite_dist

    # sanity check
    for dist in list(closest_distances_A.values()) + list(closest_distances_B.values()):
        if dist == float("Inf"):
            logger.warning(
                "Could not find closest bombsite distances for at least one node."
            )

    # collate to tuple and return
    return closest_distances_A, closest_distances_B


def _distance_internal(map_name, area_a, area_b, logger=None):
    # Use Area Distance Matrix if available, since it is faster
    # distance matrix uses strings as key
    area_a_str = str(area_a)
    area_b_str = str(area_b)
    if (
        map_name in AREA_DIST_MATRIX
        and area_a_str in AREA_DIST_MATRIX[map_name]
        and area_b_str in AREA_DIST_MATRIX[map_name][area_a_str]
    ):
        current_bombsite_dist = AREA_DIST_MATRIX[map_name][area_a_str][area_b_str][
            "geodesic"
        ]
    # Else: calculate distance from pairwise iteration over all areas in map
    else:
        if logger and logger.isEnabledFor(logging.DEBUG) and len(AREA_DIST_MATRIX) > 0:
            logger.debug("Area matrix exists but does not contain areaid: %d" % area_a)
        geodesic_path = area_distance(
            map_name=map_name, area_a=area_a, area_b=area_b, dist_type="geodesic"
        )
        current_bombsite_dist = geodesic_path["distance"]
    return current_bombsite_dist


async def process_single_demo(demo_path, queue=None, key=None):
    # logger
    uuid = Path(demo_path).stem
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path("research_project/graphs") / uuid / "logs" / f"{timestamp}.log"
    logger = get_logger(
        log_path, name=f"create_graphs_logger_{uuid}", level=logging.DEBUG
    )

    dm = DataManager(Path(demo_path), do_validate=False, logger=logger)
    output_folder = Path(__file__).parent / "../graphs/" / Path(demo_path).stem
    output_filename_template = str(output_folder / "graph-rounds-%d.pkl")
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Processing match id: %s with %d rounds."
        % (dm.get_match_id(), dm.get_round_count())
    )

    # load the labels for the match
    strategy_labels = {}
    tactic_path = (
        Path.cwd()
        / "research_project"
        / "tactic_labels"
        / dm.get_map_name()
        / f"{dm.get_match_id()}.json"
    )
    if not tactic_path.exists():
        logger.warning(
            "No tactic labels found for match %s. Using default label 'unknown'."
            % dm.get_match_id()
        )
    else:
        logger.info("Loading tactic labels from %s." % tactic_path)
        # load the labels for the match
        with open(tactic_path, "r") as f:
            strategy_labels = json.load(f)

    start_time = time.time()
    total_frames = len(dm.get_all_frames())
    processed_frames = 0
    graphs_total = 0
    for round_idx in range(dm.get_round_count()):
        output_filename = output_filename_template % round_idx
        logger.info("Converting round %d to file %s." % (round_idx, output_filename))

        progress = round((processed_frames / total_frames) * 100, 2)
        eta = dm.get_estimated_finish(
            start_time=start_time, processed_frames=processed_frames
        )
        await send_progress_embed(
            progress=progress,
            roundsTotal=dm.get_round_count(),
            currentRound=round_idx,
            eta=eta,
            id=dm.get_match_id(),
            sendSilent=(
                round_idx not in [0, dm.get_round_count() - 1]
            ),  # Send silent for first and last round
            logger=logger,
        )

        # we need to swap mappings, because player sides switch here.
        # WARNING: This only works if teams player in MR15 setting.
        if round_idx == 15:
            dm.swap_player_mapping()

        # process round
        round_label = strategy_labels.get(str(round_idx + 1), "unknown")
        graphs = process_round(
            dm,
            round_idx,
            strategy_used=round_label,
            queue=queue,
            key=key,
            logger=logger,
        )
        with open(output_filename, "wb") as f:
            pickle.dump(graphs, f)

        logger.info("%d graphs written to file." % len(graphs))
        graphs_total += len(graphs)
        processed_frames += len(graphs)

    logger.info("✅ SUCCESSFULLY COMPLETED: %d graphs written in total." % graphs_total)


def process_single_demo_sync(demo_path, queue, key):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_single_demo(demo_path, queue=queue, key=key))


def progress_monitor(queue, total_map):
    pbars = {
        k: tqdm(total=v, desc=k, position=i, leave=True)
        for i, (k, v) in enumerate(total_map.items())
    }
    finished = set()
    while len(finished) < len(pbars):
        task = queue.get()
        if task is None:
            break
        key, n = task
        if key in pbars:
            pbars[key].update(n)
            if pbars[key].n >= pbars[key].total:
                finished.add(key)
    for pbar in pbars.values():
        pbar.close()


async def main():
    demo_filenames = get_demo_files_from_list("file_paths.json", compressed=False)

    demo_pathnames = [
        "research_project/demos/dust2/" + demo_filename
        for demo_filename in demo_filenames
    ]

    batch_size = 10

    # Calculate total frames per demo for progress bars
    total_map = {
        demo: len(DataManager(Path(demo), do_validate=False).get_all_frames())
        for demo in demo_pathnames[:batch_size]
    }
    manager = Manager()
    queue = manager.Queue()
    monitor = Process(target=progress_monitor, args=(queue, total_map))
    monitor.start()

    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                functools.partial(process_single_demo_sync, demo, queue, demo),
            )
            for demo in demo_pathnames[:batch_size]
        ]
        await asyncio.gather(*tasks)

    queue.put(None)
    monitor.join()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
