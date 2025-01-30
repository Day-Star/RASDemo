__credits__ = ["Gabriel Chenevert"]
__license__ = "MIT"

from os import path
from typing import Optional
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils

class CombineEnv(gym.Env):

    """
    The RASCombineEnv class represents an environment for a tractor unloading 
    a combine harvester. The tractor must drive beside the combine and remain
    there while the combine continues to harvest grain. The tractor must not
    collide with the combine or drive over unharvested crops.

    ## Coordinate System

    The tractor and combine both exist in a 2D coordinate system. 
    
    - `x` : The axis parallel to the trajectory of the combine.
    - `y` : The axis perpendicular to the trajectory of the combine. The combine moves along y=0.

    ## Observation Space

    The observation space is a `ndarray` with shape `(5,)`. The observations in x and y are made relative to the combine.

    - `x` : The distance between the tractor and the combine along the x-axis.
    - `y` : The distance between the tractor and the combine along the y-axis.
    - `theta` : The angle of the tractor.
    - `vt` : The speed of the tractor.
    - `vc` : The speed of the combine.

    ## Control Space

    The action space is a `ndarray` with shape `(2,)`. All actions are taken by the tractor.

    - `thetaDot` : The angular velocity of the tractor.
    - `at` : The acceleration of the tractor.

    ## Disturbance Space

    The combine may accelerate erratically, causing the tractor to have to adjust its speed to keep up.

    - `ac` : The acceleration of the combine.

    ## Reward

    The reward is based on both reach ad avoid objectives.

    - `reach` : The tractor is rewarded for being within a .5 unit radius of a point 1.2m below the combine.
    - `avoid` : The tractor must avoid the area in front of the combine, above the combine, and within a .8m radius of the combine.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, hPolicy = None, do_term = False, term_reward = None, render_mode: Optional[str] = None):
        """
        Initializes the environment.

        Args:
            hPolicy: the critic policy used for training the V policy. If `None`, the environment is in H policy mode.
            render_mode: the rendering mode for the environment. Can be "human" or "rgb_array".
        """

        # Set the rendering mode
        self.render_mode = render_mode

        # Set the time step
        self.dt = 0.1

        # Set the tractor parameters
        self.maxTractorSpeed = 3.0  # Maximum speed of the tractor m/s
        self.maxThetaDot = 85.0     # Maximum angular velocity of the tractor deg/s
        self.maxTractorAccel = 3.0  # Maximum acceleration of the tractor m/s^2
        self.tractorRadius = .4     # Radius of the tractor in meters

        # Set the combine parameters
        self.maxCombineSpeed = 2.0  # Maximum speed of the combine m/s
        self.minCombineSpeed = 0.5  # Minimum speed of the combine m/s
        self.maxCombineAccel = 1.5  # Maximum acceleration of the combine m/s^2
        self.combineRadius = .4     # Radius of the combine in meters

        # Set the field parameters
        self.fieldWidth = 10.0      # Width of the field in meters
        self.fieldHeight = 10.0     # Height of the field in meters
        self.combineOffset = 2.0    # Offset of the combine from the top of the field in meters

        # Set the target parameters
        self.targetRadius = .5      # Radius of the target in meters
        self.targetOffset = 1.2     # Offset of the target from the combine in meters

        # Set the observation space
        high = np.array([self.fieldWidth/2, self.combineOffset, 180, self.maxTractorSpeed, self.maxCombineSpeed], dtype=np.float64)
        low = np.array([-self.fieldWidth/2, (self.combineOffset - self.fieldHeight), -180, -self.maxTractorSpeed, -self.minCombineSpeed], dtype=np.float64)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # Set the control space
        high = np.array([self.maxThetaDot, self.maxTractorAccel], dtype=np.float64)
        low = np.array([-self.maxThetaDot, -self.maxTractorAccel], dtype=np.float64)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # Set the disturbance space
        high = np.array([self.maxCombineAccel], dtype=np.float64)
        low = np.array([-self.maxCombineAccel], dtype=np.float64)
        self.disturbance_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # Set the H policy
        self.hPolicy = hPolicy

        # Flag if we have the h policy
        self.V_train = True if hPolicy is not None else False

        # Set the termination parameters
        self.do_term = do_term

        # Set the termination reward
        self.term_reward = term_reward

    def g_x(self, x, y):
        """
        The g_x function is the reach reward function for the environment.

        Args:
            x: the x coordinate of the tractor.
            y: the y coordinate of the tractor.

        Returns:
            The reach reward.
        """

        # Calculate the distance to the target
        return self.targetRadius - np.sqrt(x**2 + (y - self.targetOffset)**2)
    
    def l_x(self, x, y):
        """
        The l_x function is the avoid reward function for the environment.

        Args:
            x: the x coordinate of the tractor.
            y: the y coordinate of the tractor.

        Returns:
            The avoid reward.
        """

        # Crops above the combine
        l1 = -y -self.tractorRadius + 2 * self.combineRadius

        # Crops in front of the combine
        l2 = min(abs(x-50) - 50, abs(y) - (self.combineRadius + self.tractorRadius))

        # Calculate the distance to the combine
        combine_dist = np.sqrt(x**2 + y**2) - (self.combineRadius + self.tractorRadius)

        # Return the minimum of the three
        return min(l1, l2, combine_dist)
    
    def step(self, u):
        """
        Updates the environment state with the given control input.

        Args:
            u: the combined control and disturbance input.

        Returns:
            ndarray: the observation of the environment.
            float: the reward of the environment.
            bool: whether the episode is done.
            bool: whether the episode is successful.
            dict: Reach and avoid data
        """

        # Decompose the state
        x, y, theta, vt, vc = self.state

        # Decompose the control input
        thetaDot, at, ac = u

        # Allocate h_reward
        h_reward = -1

        # Spinning up a critic network if we are training the V function
        if self.V_train:

            # Generate a torch tensor from the state
            tmp_obs = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)

            # Get the control from the H network
            temp_u = self.hPolicy.control(tmp_obs)

            # Get the disturbance from the H network
            temp_d = self.hPolicy.disturbance(tmp_obs)

            # Concatenate the control and disturbance
            h_u = torch.cat((temp_u[0], temp_d[0]), 1)

            # Evaluate the critic network
            h_reward = self.hPolicy.critic(tmp_obs, h_u)

            # Extract the value
            h_reward = h_reward.item()

        # Get reach reward
        reach = self.g_x(x, y) if (not self.V_train) | (h_reward >=0) else h_reward

        # Get avoid reward
        avoid = self.l_x(x, y)

        # Calculate the reward
        reward = min(reach, avoid)

        # Compute dynamics
        theta = angleNorm(theta + thetaDot * self.dt)
        vt = np.clip(vt + at * self.dt, -self.maxTractorSpeed, self.maxTractorSpeed)
        vc = np.clip(vc + ac * self.dt, self.minCombineSpeed, self.maxCombineSpeed)
        x = x + vt * np.cos(theta) * self.dt - (vc * self.dt)
        y = y + vt * np.sin(theta) * self.dt

        # Update the state
        self.state = np.array([x, y, theta, vt, vc], dtype=np.float64)

        # Apply termination condition
        if self.do_term & (np.abs(x) > self.fieldWidth/2 or np.abs(y) > self.fieldHeight):
            
            # We are terminating, set flags and reward
            done = True
            reward = self.term_reward if self.term_reward is not None else reward

            # Return terminated
            return self._get_obs(), reward, done, False, {'reach': reach, 'avoid': avoid}

        # Check if we are rendering
        if self.render_mode == 'human':
            self.render()

        # Return the observation, reward, and done flag
        return self._get_obs(), reward, False, False, {'reach': reach, 'avoid': avoid}
    
    def reset(self, state_init = None, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment state.

        Args:
            state_init: the initial state of the environment. If `None`, the state is randomly initialized.

        Returns:
            ndarray: the observation of the environment.
        """

        # Set the random seed
        if seed is not None:
            np.random.seed(seed)

        # Check if we were given an initial state
        if state_init is not None:
            self.state = state_init
        else:
            # Initialize the state
            x = np.random.uniform(-self.fieldWidth/2, self.fieldWidth/2)
            y = np.random.uniform(-self.fieldHeight, self.combineOffset)
            theta = np.random.uniform(-180, 180)
            vt = np.random.uniform(-self.maxTractorSpeed, self.maxTractorSpeed)
            vc = np.random.uniform(self.minCombineSpeed, self.maxCombineSpeed)

            # Apply the initial state
            self.state = np.array([x, y, theta, vt, vc], dtype=np.float64)

        # Return the observation
        return self._get_obs(), {}
    
    def _get_obs(self):
        """
        Gets the observation of the environment.

        Returns:
            ndarray: the observation of the environment.
        """

        # Unpack the state
        x, y, theta, vt, vc = self.state

        # Repack and return the observation
        return np.array([x, y, theta, vt, vc], dtype=np.float64)
    
    def render(self):
        """
        Renders the environment.
        """

        # You must have a render mode selected
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        
        # Render the environment
        
            

def angleNorm(self, theta):
    """
    Normalizes the angle to be between -180 and 180 degrees.

    Args:
        theta: the angle to normalize.

    Returns:
        The normalized angle.
    """

    return (theta + 180) % 360 - 180