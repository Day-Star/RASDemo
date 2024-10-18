from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RASSwarmEnv(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster 
    def __init__(self):
        self.render_mode = None
        self.dt = 0.1
        self.high = np.array([
            3, 3, 1,
            1, 1, 1,
            3, 3, 1,
            1, 1, 1,
            3, 3, 1,
            1, 1, 1,
        ])
        self.low = np.array([
            -3,-3,-1,
            -1, -1, -1,
            -3,-3,-1,
            -1, -1, -1,
            -3,-3,-1,
            -1, -1, -1,
        ])
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action1_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) # control action space
        self.action2_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) # control action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32) # joint action space

    def step(self, action):
        K2 = np.array([ 9.1704,   16.8205])
        ego_drone = self.state[:6]
        
        target1 = np.array([1.5, 1.5, 0., 0., 0., 0.])
        target2 = np.array([-1.5, 1., 0., 0., 0., 0.])
        rew = (0.5 - np.linalg.norm(ego_drone,2))
        constraint = min([
            # np.sqrt((ego_drone[0]-ad_drone1[0])**2 + (ego_drone[1]-ad_drone1[1])**2 + (ego_drone[2]-ad_drone1[2])**2) - 0.2, # drone 1
            # np.sqrt((ego_drone[0]-ad_drone2[0])**2 + (ego_drone[1]-ad_drone2[1])**2 + (ego_drone[2]-ad_drone2[2])**2) - 0.2, # drone 2
            np.sqrt(((ego_drone[0]-1.5))**2 + ((ego_drone[1]))**2 + 0*(ego_drone[2])**2) - 0.3, # fence 1
            # np.sqrt(((ego_drone[0]+1))**2 + ((ego_drone[1]))**2 + 0*(ego_drone[2])**2) - 0.2, # fence 1
            (np.sqrt(((ego_drone[0]-self.state[6]))**2 + ((ego_drone[1]-self.state[7]))**2 + 0*(ego_drone[2] - self.state[8])**2) - 0.3), # drone 1
            (np.sqrt(((ego_drone[0]-self.state[12]))**2 + ((ego_drone[1]-self.state[13]))**2 + 0*(ego_drone[2]-self.state[14])**2) - 0.3), # drone 2
        ])
        x = self.state
        act1 = np.array([
            K2@np.array([target1[0]-x[6], target1[3]-x[9]]),
            K2@np.array([target1[1]-x[7], target1[4] - x[10]]),
            K2@np.array([target1[2]-x[8], target1[5] - x[11]]),
        ])
        act2 = np.array([
            K2@np.array([target2[0]-x[6+6], target2[3]-x[9+6]]),
            K2@np.array([target2[1]-x[7+6], target2[4] - x[10+6]]),
            K2@np.array([target2[2]-x[8+6], target2[5] - x[11+6]]),
        ])
        self.state = self.state + self.dt*np.array([
            x[3], x[4], x[5],
            1.1*action[0] + 0.1* action[3], 1.1*action[1] + 0.1*action[4], 1.1*action[2] + 0.1 * action[5],
            x[9], x[10], x[11],
            act1[0], act1[1], act1[2],
            x[15], x[16], x[17],
            act2[0], act2[1], act2[2],
        ])
        done = False
        if any(self.state[:6] > self.high[:6]) or any(self.state[:6] < self.low[:6]):
            done = True
        info = {"constraint": np.array([constraint*5]).astype(np.float32)}
        return self.state.astype(np.float32), rew*5, done, False, info
    
    def reset(self, initial_state=None,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if initial_state is None:
            low = np.array([
                -1., -1., -0.8,
                -0.8, -0.8, -0.8,
                1, 1, -1,
                -0.5, -0.5, -0.5,
                -3, -3, -1,
                -0.5, -0.5, -0.5,
            ])
            high = np.array([
                1., 1., 0.8,
                0.8, 0.8, 0.8,
                3, 3, 1,
                0.5, 0.5, 0.5,
                -1, 3, 1,
                0.5, 0.5, 0.5,
            ])
            self.state = np.random.uniform(low, high)
        else:
            self.state = initial_state        
        return self.state, {}