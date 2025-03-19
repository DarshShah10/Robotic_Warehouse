import math

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, spaces

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from definitions import Action


class FlattenAgents(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = env.num_agvs + env.num_pickers  # Total number of agents
        
        # Flatten Action Space
        if isinstance(env.action_space, spaces.Tuple):
            # If all spaces are Discrete, combine into a MultiDiscrete
            if all(isinstance(space, spaces.Discrete) for space in env.action_space.spaces):
                # Get the dimensions of each Discrete space
                dims = [space.n for space in env.action_space.spaces]
                self.action_space = spaces.MultiDiscrete(dims)
            else:
                raise ValueError("Unsupported action space for flattening.")
        else:
            self.action_space = env.action_space  # Keep original if not a Tuple
        
        # Flatten Observation Space
        if isinstance(env.observation_space, spaces.Tuple):
            # Get the flattened shapes for each agent's observation
            flat_shapes = [spaces.flatdim(space) for space in env.observation_space.spaces]
            total_dim = sum(flat_shapes)
            
            # Determine the low and high bounds
            low_values = []
            high_values = []
            
            for space in env.observation_space.spaces:
                if isinstance(space, spaces.Box):
                    low_values.append(np.full(spaces.flatdim(space), space.low.min()))
                    high_values.append(np.full(spaces.flatdim(space), space.high.max()))
                else:
                    # For non-Box spaces, use default bounds
                    low_values.append(np.full(spaces.flatdim(space), -float('inf')))
                    high_values.append(np.full(spaces.flatdim(space), float('inf')))
            
            low = np.concatenate(low_values)
            high = np.concatenate(high_values)
            
            self.observation_space = spaces.Box(
                low=low, 
                high=high, 
                dtype=np.float32
            )
        else:
            self.observation_space = env.observation_space  # Keep original if not a Tuple

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        
        # Handle both return types (obs only or obs + info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}
        
        # Flatten observation if it's a tuple
        if isinstance(obs, tuple):
            flattened_obs = np.concatenate([
                spaces.flatten(s, o) 
                for s, o in zip(self.env.observation_space.spaces, obs)
            ])
            return flattened_obs, info
        
        return obs, info

    def step(self, action):
        # Unflatten action if necessary
        if isinstance(self.env.action_space, spaces.Tuple):
            # For MultiDiscrete, split into individual actions
            if isinstance(action, np.ndarray):
                # Convert flattened action into a tuple of individual actions
                unflattened_action = []
                start_idx = 0
                
                for space in self.env.action_space.spaces:
                    if isinstance(space, spaces.Discrete):
                        # For Discrete, just take one value
                        unflattened_action.append(int(action[start_idx]))
                        start_idx += 1
                    else:
                        # For other spaces (if needed)
                        raise ValueError("Unsupported action space type")
                
                action = tuple(unflattened_action)
        
        # Call the environment's step method
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Flatten observation if it's a tuple
        if isinstance(obs, tuple):
            obs = np.concatenate([
                spaces.flatten(s, o) 
                for s, o in zip(self.env.observation_space.spaces, obs)
            ])
        
        # Convert list/tuple of termination flags to single boolean
        if isinstance(terminated, (list, tuple)):
            terminated = any(terminated)
        if isinstance(truncated, (list, tuple)):
            truncated = any(truncated)
        
        # Sum rewards if it's a list (this is common for multi-agent environments)
        if isinstance(reward, (list, tuple)):
            reward = sum(reward)
        
        return obs, reward, terminated, truncated, info
    
class DictAgents(gym.Wrapper):
    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        digits = int(math.log10(self.n_agents)) + 1

        return {f"agent_{i:{digits}}": obs_i for i, obs_i in enumerate(observation)}

    def step(self, action):
        digits = int(math.log10(self.n_agents)) + 1
        keys = [f"agent_{i:{digits}}" for i in range(self.n_agents)]
        assert keys == sorted(action.keys())

        # unwrap actions
        action = [action[key] for key in sorted(action.keys())]

        # step
        observation, reward, terminated, truncated, info = super().step(action)

        # wrap observations, rewards,  terminated and truncated
        observation = {
            f"agent_{i:{digits}}": obs_i for i, obs_i in enumerate(observation)
        }
        reward = {f"agent_{i:{digits}}": rew_i for i, rew_i in enumerate(reward)}
        terminated = {f"agent_{i:{digits}}": terminated_i for i, terminated_i in enumerate(terminated)}
        truncated = {f"agent_{i:{digits}}": truncated_i for i, truncated_i in enumerate(truncated)}
        truncated["__all__"] = all(truncated.values())

        return observation, reward, terminated, truncated, info


class FlattenSAObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""
    def __init__(self, env):
        super(FlattenSAObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [spaces.Box(low=-float('inf'), high=float('inf'), shape=(flatdim,), dtype=np.float32)]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return [spaces.flatten(obs_space, obs) for obs_space, obs in zip(self.env.observation_space, observation)]

class SquashDones(gym.Wrapper):

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, all(done), info
