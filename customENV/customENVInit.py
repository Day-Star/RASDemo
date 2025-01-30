
from gymnasium.envs.registration import register

"""
Custom Environment Registration
This file should be imported in any script that attempts to initialize a custom environment.
It adds all environments found in the customENV folder to the gym registry.
"""

register(
    id= "RASCart-v1",  # Unique identifier for the environment
    entry_point="customENV.RASCart:RASCartEnv",  # Entry point for the environment class
    max_episode_steps=500,  # Maximum number of steps allowed in an episode
    reward_threshold=-250.0,  # Threshold for the reward function
)

register(
    id= "RASFly-v1",  # Unique identifier for the environment
    entry_point="customENV.RASFly:RASFlyEnv",  # Entry point for the environment class
    max_episode_steps=200,  # Maximum number of steps allowed in an episode
    reward_threshold= 500.0,  # Threshold for the reward function
)

register(
    id= "RASSwarm-v1",  # Unique identifier for the environment
    entry_point="customENV.RASSwarm:RASSwarmEnv",  # Entry point for the environment class
    max_episode_steps=300,  # Maximum number of steps allowed in an episode
    reward_threshold= 500.0,  # Threshold for the reward function
)

register(
    id= "DroneChase-v1",  # Unique identifier for the environment
    entry_point="customENV.DroneChase:DroneChaseEnv",  # Entry point for the environment class
    max_episode_steps=2000,  # Maximum number of steps allowed in an episode
    reward_threshold= 500.0,  # Threshold for the reward function
)

register(
    id= "DroneChase-v2",  # Unique identifier for the environment
    entry_point="customENV.DroneChase:DroneChaseEnv",  # Entry point for the environment class
    max_episode_steps=700,  # Maximum number of steps allowed in an episode
    reward_threshold= 500.0,  # Threshold for the reward function
)

register(
    id= "Combine-v1",  # Unique identifier for the environment
    entry_point="customENV.RASCombine.CombineEnv",  # Entry point for the environment class
    max_episode_steps=1000,  # Maximum number of steps allowed in an episode
    reward_threshold= 500.0,  # Threshold for the reward function
)