# register.py
import gymnasium as gym
from warehouse import Warehouse  # Import your modified Warehouse
from definitions import RewardType
from spaces import observation_map # Import observation maps

_obs_types = list(observation_map.keys())

_sizes = {
    "tiny": (1, 3),
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
    "extralarge": (4, 7),
}

_request_queues = {
    "tiny": 20,
    "small": 20,
    "medium": 20,
    "large": 40,
    "extralarge": 60,
}
import itertools
_perms = itertools.product(_sizes.keys(), _obs_types, range(1,20), range(1, 10))
for size, obs_type, num_agvs, num_pickers in _perms:
    gym.register(
        id=f"tarware-{size}-{num_agvs}agvs-{num_pickers}pickers-{obs_type}obs-v1",
        entry_point="warehouse:Warehouse",  # Correct entry point
        kwargs={
            "column_height": 8,
            "shelf_rows": _sizes[size][0],
            "shelf_columns": _sizes[size][1],
            "num_agvs":  num_agvs,
            "num_pickers": num_pickers,
            "request_queue_size": _request_queues[size],
            "max_inactivity_steps": None,
            "max_steps": 500,
            "reward_type": RewardType.INDIVIDUAL,
            "observation_type": obs_type,
        },
    )


def full_registration():
    _perms = itertools.product(_sizes.keys(), _obs_types, _request_queues, range(1,20), range(1, 10),)
    for size, obs_type, num_agvs, num_pickers in _perms:
        # normal tasks with modified column height
        gym.register(
            id=f"tarware-{size}-{num_agvs}agvs-{num_pickers}pickers-{obs_type}obs-v1",
            entry_point="warehouse:Warehouse",
            kwargs={
                "column_height": 8,
                "shelf_rows": _sizes[size][0],
                "shelf_columns": _sizes[size][1],
                "num_agvs":  num_agvs,
                "num_pickers": num_pickers,
                "sensor_range": 1,
                "request_queue_size": _request_queues[size],
                "max_inactivity_steps": None,
                "max_steps": 500,
                "reward_type": RewardType.INDIVIDUAL,
                "observation_type": obs_type,
            },
        )
# full_registration()  # You can uncomment this if you want the full registration