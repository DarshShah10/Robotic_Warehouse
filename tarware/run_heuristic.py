# C:\Darsh\Projects\Robotic warehouuse\task-assignment-robotic-warehouse\run_heuristic.py
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gymnasium as gym

from heuristic import heuristic_episode  # Relative import

print("Script started.")

parser = ArgumentParser(description="Run tests with vector environments on WarehouseEnv", formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument(
        "--num_episodes",
        default=1000,
        type=int,
        help="The number of episodes to run"
    )
parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="The seed to run with"
    )

parser.add_argument(
        "--render",
        action='store_true',
        help="Render the environment"
    )

args = parser.parse_args()
print(f"Arguments parsed: {args}")

def info_statistics(infos, global_episode_return, episode_returns):
    _total_deliveries = 0
    _total_clashes = 0
    _total_stuck = 0
    for info in infos:
        _total_deliveries += info["shelf_deliveries"]
        _total_clashes += info["clashes"]
        _total_stuck += info["stucks"]
        info["total_deliveries"] = _total_deliveries
        info["total_clashes"] = _total_clashes
        info["total_stuck"] = _total_stuck
    last_info = infos[-1]
    last_info["episode_length"] = len(infos)
    last_info["global_episode_return"] = global_episode_return
    last_info["episode_returns"] = episode_returns
    return last_info

if __name__ == "__main__":
    try:
        print("Creating environment...")
        # Corrected environment ID
        env = gym.make("tarware-extralarge-14agvs-7pickers-partialobs-v1")
        print("Environment created.")
        seed = args.seed
        completed_episodes = 0
        for i in range(args.num_episodes):
            print(f"Starting episode {i}...")
            start = time.time()
            obs = env.reset(seed=seed + i)
            print(f"Environment reset for episode {i}.")
            infos, global_episode_return, episode_returns = heuristic_episode(env.unwrapped, args.render)
            print(f"Heuristic episode {i} completed.")
            end = time.time()
            last_info = info_statistics(infos, global_episode_return, episode_returns)
            last_info["overall_pick_rate"] = last_info.get("total_deliveries") * 3600 / (5 * last_info['episode_length'])
            episode_length = len(infos)
            print(f"Completed Episode {completed_episodes}: | [Overall Pick Rate={last_info.get('overall_pick_rate'):.2f}]| [Global return={last_info.get('global_episode_return'):.2f}]| [Total shelf deliveries={last_info.get('total_deliveries'):.2f}]| [Total clashes={last_info.get('total_clashes'):.2f}]| [Total stuck={last_info.get('total_stuck'):.2f}] | [FPS = {episode_length/(end-start):.2f}]")
            completed_episodes += 1
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()