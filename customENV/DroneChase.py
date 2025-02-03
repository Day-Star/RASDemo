__credits__ = ["Gabriel Chenevert"]

from typing import Optional
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

# Default environmet bounds
DEFAULT_X = 3.0 # Default +/- x-axis bound
DEFAULT_Y = 3.0 # Default +/- y-axis bound
DEFAULT_Z = 3.0 # Default + z-axis bound (Note: z-axis is positive only to avoid sudden encounters with the ground)

class DroneChaseEnv(gym.Env):
    """
    
    ## Description:
        This is a custom Gym environment for a drone chasing a ground robot using the Reach Avoid Stay algorithm.

    ![Swarm Coordinate System]
    - `x` : the global x coordinate of the drone in the 3D space (meters)
    - `y` : the global y coordinate of the drone in the 3D space (meters)
    - `z` : the global z coordinate of the drone in the 3D space (meters)
    - `xdot` : the velocity of the drone in the x-axis (m/s)
    - `ydot` : the velocity of the drone in the y-axis (m/s)
    - `zdot` : the velocity of the drone in the z-axis (m/s)

    ## Action Space:
        The action space consists of four continuous actions representing the velocity 
        commands sent to the drone in the x, y, z, and theta directions. It is a `ndarray` with shape `(3,)`.

        | Num | Action | Min  | Max |
        |-----|--------|------|-----|
        | 0   |  xdot  | -1.0 | 1.0 |
        | 1   |  ydot  | -1.0 | 1.0 |
        | 2   |  zdot  | -1.0 | 1.0 |

    ## Observation Space:
        The observation space consists of the drone's current position and orientation in the 3D space and the position of the vehicle it is pursuing in 2D space.
        The height of the vehicle above the ground is fixed at .28 meters above the ground, and therefore not included in the observation space. The observation space is a `ndarray` with shape `(10,)`.

        | Num | Observation | Min  | Max |
        |-----|-------------|------|-----|
        | 0   |  x          | -1.0 | 1.0 |
        | 1   |  y          | -1.0 | 1.0 |
        | 2   |  z          |  0.0 | 2.0 |
        | 3   |  xdot       | -1.0 | 1.0 |
        | 4   |  ydot       | -1.0 | 1.0 |
        | 5   |  zdot       | -1.0 | 1.0 |
        | 6   |  vx         | -1.0 | 1.0 |
        | 7   |  vy         | -1.0 | 1.0 |
        | 8   |  vxdot      | -1.0 | 1.0 |
        | 9   |  vydot      | -1.0 | 1.0 |


    ## Rewards:

    This environment uses a combination of two rewards: reach (g(x)) and avoid (l(x)). In both cases, positive values are good, and negative values are bad.
    The minimum of the two rewards is returned as the final reward along with split outs from both rewards.
    When training the V(x) actor, the reaching reward g(x) is replaced with the trained H(x) function.

    ### Reach Reward (g(x)): 
        The reach reward is calculated based on the distance between the drone and the target position. 
        The reward is calculated as follows:
        `g(x) = (target_radius - sqrt((x - x_target)^2 + (y - y_target)^2 + (z - z_target)^2))`
        where: 
            - `target_radius` is the radius of the target position (meters)
            - `(x_target, y_target, z_target)` is the target position in the 3D space (meters)
            - `(x, y, z)` is the current position of the drone in the 3D space (meters)
            - `target_buff` is the score multiplier for reaching the target

    ## Avoid Reward (l(x)):
        The avoid reward is calculated based on the distance between the drone and the base of the target vehicle.
        The reward is calculated as follows:
        `l(x) = (sqrt((x - x_obs)^2 + (y - y_obs)^2 + (z - z_obs)^2) - obs_radius - drone_radius) * obs_buff`
        where:
            - `(x_obs, y_obs, z_obs)` is the position of the obstacle in the 3D space (meters)
            - `obs_radius` is the radius of the obstacle (meters)
            - `drone_radius` is the radius of the drone (meters)
            - `obs_buff` is the score multiplier for the obstacle

    ## Episode Truncation:
        The episode is truncated by default at 200 time steps or when the drone exceeds the environment bounds.
        This can be changed by setting the `max_episode_steps` parameter during registration.

    ## Arguments

    - `hPolicy` (torch.nn.Module): The trained H(x) function for the reach reward. Default is None.
    - `render_mode` (str): The rendering mode for the environment. Default is 'human'.

    ## Version History
    - 1.0.0: Initial release
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    def __init__(self, hPolicy = None, do_term = False, term_reward = None, xBound = None, yBound = None, zBound = None,
                  render_mode: Optional[str] = None):
        """
        Initialize a new RASSwarm environment.

        Args:
            hPolicy (torch.nn.Module): The trained H(x) function for the reach reward. Default is None.
            xBound (float): The x-axis bound of the environment. Default is 1.0.
            yBound (float): The y-axis bound of the environment. Default is 1.0.
            zBound (float): The z-axis bound of the environment. Default is 2.0.
            render_mode (str): The rendering mode for the environment. Default is 'None'.
        """

        # Setting Crazyflie performance characteristics
        self.max_speed = 1.0                # Maximum horizontal speed of the drone in m/s
        self.max_z_speed = 1.0              # Maximum vertical speed of the drone in m/s
        self.maxAcceleration = 1.0          # Maximum acceleration of the drone in m/s^2
        self.drone_radius = (150/2)/1000        # Drone radius in meters, set to 150mm actual drone is ~140mm

        # Setting environment bounds
        self.x_bound = DEFAULT_X if xBound is None else xBound  # Default +/- x-axis bound
        self.y_bound = DEFAULT_Y if yBound is None else yBound  # Default +/- y-axis bound
        self.z_bound = DEFAULT_Z if zBound is None else zBound  # Default + z-axis bound
        self.dt = 0.1                                           # Time step in seconds

        # Target and obstacle
        self.target_buff = 10               # Target score bonus
        self.target_radius = 0.5           # Target radius in meters
        self.target_z = 1                   # Target height in meters
        self.obs_buff = 10                  # Obstacle score bonus
        self.obs_radius = 0.762 / 2         # Obstacle radius in meters, set to 762mm
        self.obstacle_z = 0.28              # Obstacle height in meters
        self.v_speed = self.max_speed * .5 # Maximum speed of the vehicle in m/s
        self.v_accel = self.maxAcceleration * .5 # Maximum acceleration of the vehicle in m/s^2
        self.vx_bound = self.x_bound * .8   # Vehicle x bound
        self.vy_bound = self.y_bound * .8   # Vehicle y bound

        # Setting termination bounds
        self.t_buffer = 0

        # Setting termination parameters
        self.do_term = do_term          # Should we terminate the episode if the drone goes out of bounds?
        self.term_reward = term_reward  # Fixed reward for going out of bounds

        # Setting up hFunction
        self.hPolicy = hPolicy

        # Checking if we have a trained H(x) function and are training V
        self.V_train = True if hPolicy is not None else False

        # Put the hPolicy in evaluation mode if we have one
        if self.V_train:
            self.hPolicy.eval()

        # Setting environment render parameters
        self.render_mode = render_mode  # Rendering mode
        self.screen_dim = 500           # Screen dimension
        self.screen = None              # Not setting up a screen
        self.clock = None               # Not setting up a clock
        self.isopen = True              # Open the render window if we have one

        # Setting up observation space
        high = np.array([self.x_bound, self.y_bound, self.z_bound, self.max_speed, self.max_speed, self.max_z_speed,
                         self.x_bound, self.y_bound, self.v_speed, self.v_speed])
        low = np.array([-self.x_bound, -self.y_bound, 0, -self.max_speed, -self.max_speed, -self.max_z_speed,
                        -self.vx_bound, -self.vy_bound, -self.v_speed, -self.v_speed])
        
        # Applying the observation space
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        # Setting up control space
        high = np.array([self.maxAcceleration, self.maxAcceleration, self.maxAcceleration])

        # Applying the control space
        self.control_space = spaces.Box(-high, high, dtype=np.float64)

        # Setting up disturbance space
        high = np.array([self.v_accel, self.v_accel])

        # Applying the disturbance space
        self.disturbance_space = spaces.Box(-high, high, dtype=np.float64)

        # Setting up action space
        high = np.array([self.max_speed, self.max_speed, self.max_z_speed, self.v_accel, self.v_accel])

        # Concatinating the observation and control spaces to action space
        self.action_space = spaces.Box(-high, high, dtype=np.float64)
    
    def g_x(self, x, t):
        """
        Calculates the reach reward based on the distance between the drone and the target position.

        Args:
            x (ndarray): The current position of the drone in the 3D space.
            t (ndarray): The target position in the 3D space.

        Returns:
            float: The reach reward.
        """
        # Calculate g(x) reward
        reward = (self.target_radius - np.sqrt(np.sum((x[:6] - t)**2))) * self.target_buff

        # Check if we are in the target
        if reward > 0:

            # We are in the target, return the reward multiplied by the target buff again
            return reward * self.target_buff
        
        return reward
    
    def l_x(self, x, o):
        """
        Calculates the avoid reward based on the distance between the drone and the other drones and buildings.

        Args:
            x (ndarray): The current position of the drone in 3D space.
            o (ndarray): The positions of the obstacle in 3D space.

        Returns:
            float: The avoid reward.
        """

        # Returning the sum of the avoid rewards
        return (np.sqrt(np.sum((x[:3] - o)**2)) - self.obs_radius - self.drone_radius) * self.obs_buff
    
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

        # Getting last action for rendering
        self.last_u = u

        # Decomposing state
        x, y, z, xdot, ydot, zdot, vx, vy, vxdot,  vydot = self.state

        # Decomposing action
        xu, yu, zu, vxu, vyu = u

        # Create a target position
        target = np.array([vx, vy, self.target_z, vxdot,0,0])

        # Create obstacle position
        obstacle = np.array([vx, vy, self.obstacle_z])

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

        # Calculate Reward

        # Calculate reach reward
        reach = self.g_x(self.state,target) if (not self.V_train) | (h_reward>= 0) else h_reward

        # Calculate avoid reward
        avoid = self.l_x(self.state, obstacle)

        # Report worst case reward
        reward = min(reach, avoid)

        # Update position
        x = x + xdot * self.dt
        y = y + ydot * self.dt
        z = z + zdot * self.dt

        # Update velocity
        xdot = np.clip(xdot + xu * self.dt, -self.max_speed, self.max_speed)
        ydot = np.clip(ydot + yu * self.dt, -self.max_speed, self.max_speed)
        zdot = np.clip(zdot + zu * self.dt, -self.max_z_speed, self.max_z_speed)

        # Update vehicle

        # Vehicle position
        vx = np.clip(vx + vxdot * self.dt, -self.vx_bound, self.vx_bound)
        vy = np.clip(vy + vydot * self.dt, -self.vy_bound, self.vy_bound)

        # Vehicle velocity
        vxdot = np.clip(vxdot + vxu * self.dt, -self.v_speed, self.v_speed)
        vydot = np.clip(vydot + vyu * self.dt, -self.v_speed, self.v_speed)

        # Update state
        self.state = np.array([x, y, z, xdot, ydot, zdot, vx, vy, vxdot, vydot], dtype=np.float64)

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
    
    def reset(self, state_init = None, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to a random state.

        Args:
            state_init (ndarray): The initial state of the environment. The initial state should only be [x,y,z,xdot,ydot,zdot] for the agent drone. Default is None.
            time_init (float): The initial time of the environment. Default is None.
            seed (int): The seed to use for the random number generator. Default is None.
            options (dict): Additional options for the reset. Default is None.

        Returns:
            ndarray: The initial observation of the environment.
        """

        # Applying seed
        if seed is not None:
            np.random.seed(seed)

        # Checking if we have an initial state
        if state_init is not None:

            # Unpack initial state
            [x,y,z,xdot,ydot,zdot,vx,vy,vxdot,vydot] = state_init

            # Set the state
            self.state = np.array([x, y, z, xdot, ydot, zdot, vx, vy, vxdot, vydot], dtype=np.float64)

        else:

            # We do not, set the state to a random position within the bounds

            if self.V_train:

                # Set the upper bounds for the state
                high = np.array([self.x_bound, self.y_bound, self.z_bound, self.max_speed, self.max_speed, self.max_z_speed, self.vx_bound, self.vy_bound, self.v_speed, self.v_speed])

                # Set the lower bounds for the state
                low = np.array([-self.x_bound, -self.y_bound, 0, -self.max_speed, -self.max_speed, -self.max_z_speed, -self.vx_bound, -self.vy_bound, -self.v_speed, -self.v_speed])

                # Generate a random state
                [x,y,z,xdot,ydot,zdot,vx,vy,vxdot,vydot] = np.random.uniform(low, high)

            else:

                # Generate a random vehicle position
                vx = np.random.uniform(-self.vx_bound, self.vx_bound)
                vy = np.random.uniform(-self.vy_bound, self.vy_bound)
                vxdot = np.random.uniform(-self.v_speed, self.v_speed)
                vydot = np.random.uniform(-self.v_speed, self.v_speed)

                # Generate a random drone position within 2m of the vehicle
                x = np.random.uniform(max(vx - 2, self.x_bound), min(vx + 2, -self.x_bound))
                y = np.random.uniform(max(vy - 2, self.y_bound), min(vy + 2, -self.y_bound))
                z = np.random.uniform(0, self.z_bound)

                # Generate a random drone velocity
                xdot = np.random.uniform(-self.max_speed, self.max_speed)
                ydot = np.random.uniform(-self.max_speed, self.max_speed)
                zdot = np.random.uniform(-self.max_z_speed, self.max_z_speed)

            # Set the state
            self.state = np.array([x, y, z, xdot, ydot, zdot, vx, vy, vxdot, vydot], dtype=np.float64)

        # Calling render mode
        if self.render_mode == 'human':
            self.render()

        # Return the observation
        return self._get_obs(), {}
    
    def _get_obs(self):
        """
        Gets the observation of the environment.

        Returns:
            ndarray: The observation of the environment.
        """

        # Unpack the state
        x, y, z, xdot, ydot, zdot, vx, vy, vxdot, vydot = self.state

        # Repack and return
        return np.array([x, y, z, xdot, ydot, zdot, vx, vy, vxdot, vydot], dtype=np.float64)
    
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
