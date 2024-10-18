__credits__ = ["Gabriel Chenevert"]

from os import path
from typing import Optional
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

# Default environment bounds
DEFAULT_X = 1.0 # Default +/- x-axis bound
DEFAULT_Y = 1.0 # Default +/- y-axis bound
DEFAULT_Z = 2.0 # Default + z-axis bound (Note: z-axis is positive only to avoid sudden encounters with the ground)

class RASFlyEnv(gym.Env):
    """
    ## Description:
        The RASFly environment is a simple 3D flying environment that simulates a drone flying in a 3D space. 
        The goal of the agent is to learn how to fly the drone in the 3D space without crashing into obstacles or the ground.

    ![Fly Coordinate System]

    - `x` : the x coordinate of the drone in the 3D space (meters)
    - `y` : the y coordinate of the drone in the 3D space (meters)
    - `z` : the z coordinate of the drone in the 3D space (meters)
    - `xdot` : the velocity of the drone in the x-axis (m/s)
    - `ydot` : the velocity of the drone in the y-axis (m/s)
    - `zdot` : the velocity of the drone in the z-axis (m/s)

    ## Action Space:
        The action space consists of four continuous actions representing the velocity 
        commands sent to the drone in the x, y, and z directions. It is a `ndarray` with shape `(3,)`.

        | Num | Action | Min  | Max |
        |-----|--------|------|-----|
        | 0   |  xdot  | -1.0 | 1.0 |
        | 1   |  ydot  | -1.0 | 1.0 |
        | 2   |  zdot  | -1.0 | 1.0 |


    ## Observation Space:
        The observation space consists of the drone's current position and orientation in the 3D space. 
        It is a `ndarray` with shape `(6,)`.

        | Num | Observation | Min  | Max |
        |-----|-------------|------|-----|
        | 0   |     x       | -1.0 | 1.0 |
        | 1   |     y       | -1.0 | 1.0 |
        | 2   |     z       |  0.0 | 2.0 |
        | 3   |   xdot      | -1.0 | 1.0 |
        | 4   |   ydot      | -1.0 | 1.0 |
        | 5   |   zdot      | -1.0 | 1.0 |

    ## Rewards:

    This environment uses a combination of two rewards: reach (g(x)) and avoid (l(x)). In both cases, positive values are good, and negative values are bad.
    The minimum of the two rewards is returned as the final reward along with split outs from both rewards.
    When training the V(x) actor, the reaching reward g(x) is replaced with the trained H(x) function.

    ### Reach Reward (g(x)): 
        The reach reward is calculated based on the distance between the drone and the target position. 
        The reward is calculated as follows:
        `g(x) = target_radius - sqrt((x - x_target)^2 + (y - y_target)^2 + (z - z_target)^2)`
        where: 
            - `target_radius` is the radius of the target position (meters)
            - `(x_target, y_target, z_target)` is the target position in the 3D space (meters)
            - `(x, y, z)` is the current position of the drone in the 3D space (meters)
    
    ### Avoid Reward (l(x)):
        The avoid reward is calculated based on the distance between the drone and the obstacle positions. 
        The reward is calculated as follows:
        `l(x) = sqrt((x - x_obstacle)^2 + (y - y_obstacle)^2 + (z - z_obstacle)^2) - obstacle_radius`
        where:
            - `obstacle_radius` is the radius of the obstacle position (meters)
            - `(x_obstacle, y_obstacle, z_obstacle)` is the obstacle position in the 3D space (meters)
            - `(x, y, z)` is the current position of the drone in the 3D space (meters)

    ## Episode Truncation:
        The episode is truncated by default at 200 time steps or when the drone exceeds the environment bounds by 2 meters.
        This can be changed by setting the `max_episode_steps` parameter during registration.

    ## Arguments

    - `hPolicy` (torch.nn.Module): The trained H(x) function for the reach reward. Default is None.
    - `render_mode` (str): The rendering mode for the environment. Default is 'human'.

    ## Version History
    - 1.1.0: Initial release
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    def __init__(self, hPolicy = None, do_term = False, term_reward = None, xBound = None, yBound = None, zBound = None, 
                  render_mode: Optional[str] = None):
        """
        Initializes a new RASFly environment.

        Args:
            hPolicy (torch.nn.Module): The trained H(x) function for the reach reward. Default is None.
            xBound (float): The x-axis bound of the environment. Default is 1.0.
            yBound (float): The y-axis bound of the environment. Default is 1.0.
            zBound (float): The z-axis bound of the environment. Default is 2.0.
            render_mode (str): The rendering mode for the environment. Default is None.
        """
        
        # Setting Crazyflie performance characteristics
        self.max_speed = 1.0                # Maximum horizontal speed of the drone in m/s
        self.max_z_speed = 1.0              # Maximum vertical speed of the drone in m/s
        self.maxAcceleration = 0.5          # Maximum acceleration of the drone in m/s^2

        # Setting environment bounds
        self.x_bound = DEFAULT_X if xBound is None else xBound  # Default +/- x-axis bound
        self.y_bound = DEFAULT_Y if yBound is None else yBound  # Default +/- y-axis bound
        self.z_bound = DEFAULT_Z if zBound is None else zBound  # Default + z-axis bound
        self.dt = 0.1                                           # Time step in seconds

        # Setting termination bounds
        self.t_buffer = 0

        # Setting termination parameters
        self.do_term = do_term          # Should we terminate the episode if the drone goes out of bounds?
        self.term_reward = term_reward  # Fixed reward for going out of bounds

        # Calculate maximum velocity change over one time step
        self.max_speed_change = self.maxAcceleration * self.dt      # Maximum speed change in one time step
        self.max_z_speed_change = self.maxAcceleration * self.dt    # Maximum z speed change in one time step

        # Setting up hFunction
        self.hPolicy = hPolicy

        # Checking if we have a trained H(x) function and are training V
        self.V_train = True if hPolicy is not None else False

        # Put the hPolicy in evaluation mode if we have one
        if self.V_train:
            self.hPolicy.eval()

        # Defining target
        self.target = np.array([0.0, 0.0, 1.0]) # Target position
        self.target_radius = 0.1                # Target radius
        self.target_buff = 100                  # Target reward buff

        # Defining obstacle
        self.obstacle = np.array([0.5, 0.5, 1.0])   # Obstacle position
        self.obstacle_radius = 0.2                  # Obstacle radius

        # Calculating obstacle cost buff
        self.obstacle_buff = 1 #-1.5*min(self.g_x(self.obstacle + self.obstacle_radius), self.g_x(self.obstacle - self.obstacle_radius))/self.obstacle_radius

        # Setting environment render parameters
        self.render_mode = render_mode  # Rendering mode
        self.screen_dim = 500           # Screen dimension
        self.screen = None              # Not setting up a screen
        self.clock = None               # Not setting up a clock
        self.isopen = True              # Open the render window if we have one

        # Setting up observation space
        high = np.array([self.x_bound, self.y_bound, self.z_bound, 
                         self.max_speed, self.max_speed, self.max_z_speed])

        # Applying observation space
        self.observation_space = spaces.Box(low = -high, high = high, dtype=np.float64)

        # Setting environment action space
        high = np.array([self.max_speed, self.max_speed, self.max_z_speed])

        # Applying action space
        self.action_space = spaces.Box(low = -high, high = high, dtype=np.float64)
        
    def g_x(self, x):
        """
        Calculates the reach reward based on the distance between the drone and the target position.

        Args:
            x (ndarray): The current position of the drone in the 3D space.

        Returns:
            float: The reach reward.
        """

        reward = (self.target_radius - np.sqrt(np.sum((x[:3] - self.target)**2)))

        if reward > 0:
            return reward * self.target_buff
        
        return reward
    
    def l_x(self, x):
        """
        Calculates the avoid reward based on the distance between the drone and the obstacle position.

        Args:
            x (ndarray): The current position of the drone in the 3D space.

        Returns:
            float: The avoid reward.
        """

        return (np.sqrt(np.sum((x[:3] - self.obstacle)**2)) - self.obstacle_radius) * self.obstacle_buff
    
    def step(self, u):
        """
        Steps the environment using the given action.

        Args:
            u (ndarray): The action to take in the environment.

        Returns:
            ndarray: The observation of the environment after the action.
            float: The reward for the action taken.
            bool: Whether the episode has ended.
            dict: Reach and avoid data about the step.
        """

        # Clip action
        u = np.clip(u, self.action_space.low, self.action_space.high)

        # Getting last action for rendering
        self.last_u = u

        # Decomposing state
        x, y, z, xdot, ydot, zdot = self.state

        # Decomposing action
        xu, yu, zu = u

        h_reward = -1

        # Spinning up a critic network if we are training the V function
        if self.V_train:

            # Generate a torch tensor from the state
            tmp_obs = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)

            # Get the optimal control from the H network
            h_u = self.hPolicy.actor(tmp_obs)

            # Evaluate the critic network
            h_reward = self.hPolicy.critic(tmp_obs, h_u[0])

            # Extract the value
            h_reward = h_reward.item()

        # Calculate Reward

        # Calculate reach reward
        reach = self.g_x(self.state) if (not self.V_train) | (h_reward>= 0.0) else h_reward

        # Calculate avoid reward
        avoid = self.l_x(self.state)

        # Report worst case reward
        reward = min(reach, avoid)

        # Update state

        # Check if X velocity order excedes max acceleration
        if np.abs(xu - xdot) > self.max_speed_change:

            # It does, velocity update based on max speed change in the direction of order
            xdot = xdot + self.max_speed_change if xu > xdot else xdot - self.max_speed_change
        else:

            # It does not, new velocity is the commanded velocity
            xdot = xu

        # Check if Y velocity order excedes max acceleration
        if np.abs(yu - ydot) > self.max_speed_change:

            # It does, velocity update based on max speed change in the direction of order
            ydot = ydot + self.max_speed_change if yu > ydot else ydot - self.max_speed_change
        else:

            # It does not, new velocity is the commanded velocity
            ydot = yu

        # Check if Z velocity order excedes max acceleration
        if np.abs(zu - zdot) > self.max_z_speed_change:

            # It does, velocity update based on max speed change in the direction of order
            zdot = zdot + self.max_z_speed_change if zu > zdot else zdot - self.max_z_speed_change
        else:

            # It does not, new velocity is the commanded velocity
            zdot = zu

        # Clipping speed
        xdot = np.clip(xdot, -self.max_speed, self.max_speed)
        ydot = np.clip(ydot, -self.max_speed, self.max_speed)
        zdot = np.clip(zdot, -self.max_z_speed, self.max_z_speed)

        # Update position
        x = x + xdot * self.dt
        y = y + ydot * self.dt
        z = z + zdot * self.dt

        # Update state
        self.state = np.array([x, y, z, xdot, ydot, zdot], dtype=np.float64)

        # Applying termination conditions
        if self.do_term & (np.abs(x) > self.x_bound + self.t_buffer or np.abs(y) > self.y_bound + self.t_buffer or z < 0 or z > self.z_bound + self.t_buffer):
            
            # We have hit the bounds, episode is done

            if self.term_reward is not None:
                return self._get_obs(), self.term_reward, True, False, {"reach": self.term_reward, "avoid": self.term_reward}
            
            else:
                return self._get_obs(), reward, True, False, {"reach": reach, "avoid": avoid}

        # Check if we are rendering
        if self.render_mode == 'human':
            self.render()

        # Return observation, reward, done, and info
        return self._get_obs(), reward, False, False, {"reach": reach, "avoid": avoid}
    
    def reset(self, state_init = None,*, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to a random state.

        Args:
            state_init (ndarray): The initial state to use for the environment. Default is None.
            seed (int): The seed to use for the random number generator.
            options (dict): The options to use for the reset.

        Returns:
            ndarray: The initial observation of the environment.
        """

        # Applying seed
        if seed is not None:
            np.random.seed(seed)

        # Checking if we have an initial state
        if state_init is not None:

            # We do, setting state to initial state
            self.state = state_init
        else: 
            # We do not, setting state to random state
            high = np.array([self.x_bound, self.y_bound, self.z_bound, 
                             self.max_speed, self.max_speed, self.max_z_speed])
            self.state = np.random.uniform(low=-high, high=high)
            self.last_u = None
        
        # Calling render mode
        if self.render_mode == 'human':
            self.render()
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        x, y, z, xdot, ydot, zdot = self.state
        return np.array([x, y, z, xdot, ydot, zdot], dtype=np.float64)
    
    def render(self):
        """
        Renders the environment.
        """

        # You must have a render mode selected
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render mexod wixout specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # Try to setup pygame
        try:
            
            # Get pygame
            import pygame

            # Get gfxdraw
            from pygame import gfxdraw
        
        # Whelp, that didn't work...
        except ImportError as e:

            # Raise the error, we need pygame to render
            raise ImportError('To use the render mode, you must have pygame installed.') from e
            
        # Check if we have a screen
        if self.screen is None:

            # Set up the screen if human
            if self.render_mode == 'human':

                # Initialize pygame
                pygame.init()

                # Set up the screen
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
            else: # mode must be "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        
        # Set up the clock
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Create a surface
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))

        # Fill the surface with white
        self.surf.fill((255, 255, 255))

        # Convert target position to screen coordinates
        target = (self.screen_dim / 2 + (self.target[0] / self.x_bound) * self.screen_dim / 2,
                  self.screen_dim / 2 - (self.target[1] / self.y_bound) * self.screen_dim / 2)
        
        # Convert target radius
        radius = self.target_radius / self.x_bound * self.screen_dim / 2

        # Draw target
        gfxdraw.aacircle(self.surf, int(target[0]), int(target[1]), int(radius), (0, 255, 0))

        # Convert obstacle position to screen coordinates
        obstacle = (self.screen_dim / 2 + (self.obstacle[0] / self.x_bound) * self.screen_dim / 2,
                    self.screen_dim / 2 - (self.obstacle[1] / self.y_bound) * self.screen_dim / 2)
        
        # Convert obstacle radius
        radius = self.obstacle_radius / self.x_bound * self.screen_dim / 2

        # Draw obstacle
        gfxdraw.aacircle(self.surf, int(obstacle[0]), int(obstacle[1]), int(radius), (255, 0, 0))

        # Convert drone position to screen coordinates
        drone = (self.screen_dim / 2 + (self.state[0] / self.x_bound) * self.screen_dim / 2,
                 self.screen_dim / 2 - (self.state[1] / self.y_bound) * self.screen_dim / 2)
        
        # Draw drone
        gfxdraw.aacircle(self.surf, int(drone[0]), int(drone[1]), 2, (0, 0, 255))
        gfxdraw.filled_circle(self.surf, int(drone[0]), int(drone[1]), 2, (0, 0, 255))

        # Output the render
        if self.render_mode == 'human':
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()
        
        else: # mode must be "rgb_array"
            return np.transpose(pygame.surfarray.array3d(self.surf), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
