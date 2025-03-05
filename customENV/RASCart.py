__credits__ = ["Carlos Luis","Gabriel Chenevert"]
__license__ = "MIT"

from typing import Optional
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


DEFAULT_X = 2.0
DEFAULT_XDOT = 1.0


class RASCartEnv(gym.Env):
    """
    The RASCartEnv class represents an environment for a 2d cart problem. The cart moves back and forth along a 1D path with a target and an obstacle.
    The goal is to reach the target while avoiding the obstacle. 
    
    ## Description

    ![Cart Coordinate System]

    -  `x` : The x-coordinate of the cart along a 1D path in m.
    - `xdot` : Velocity of the cart in m/s.
    - `xdotdot`: Acceleration of the cart in m/s^2.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` representing the acceleration applied to the cart.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   |  Acc   | -2.0 | 2.0 |


    ## Observation Space

    The observation space is a `ndarray` with shape `(2,)` representing the x position and the velocity of the cart.

    | Num | Observation  | Min  | Max |
    |-----|--------------|------|-----|
    | 0   |       x      | -5.0 | 5.0 |
    | 1   |     xdot     | -1.0 | 1.0 |

    ## Rewards

    This environment uses a combination of two rewards, reach and avoid. For both, positive values are good and negative values are bad.
    The minimum of the two rewards is reported as the reward.

    Reach g(x): the agent receives a reward proportional to the distance from the cart to the target.
    If the cart is within the target, the reward is positive and proportional to the distance from the edge of the target.
    If the cart is outside the target, the reward is negative. By default, the target is located at 0 meters, and has a radius of 0.5 meters.
    the equation is: \[g(x) = \text{target_radius} - \sqrt{(x - \text{target})^2} = \text{target_radius} - |x - target|\]

    Avoid l(x): the agent receives a reward proportional to the distance from the cart to the obstacle.
    If the cart is within the obstacle, the reward is negative and proportional to the distance from the edge of the obstacle.
    If the cart is outside the obstacle, the reward is positive. By default, the obstacle is located at -4.8 meters, and has a radius of 0.2 meters.
    the equation is: \[l(x) = \sqrt{(x - \text{obstacle})^2} - \text{obstacle_radius} = |x - \text{obstacle}| - \text{obstacle_radius}\]

    ## Starting State

    The starting state is a random position in *[-5, 5]* and a random angular velocity in *[-1,1]*.

    ## Episode Truncation

    The episode truncates at 500 time steps.

    ## Arguments

    - `hPolicy`: the critic policy used for training the V policy. If `None`, the environment is in H policy mode.
    - `render_mode`: the rendering mode for the environment. Can be "human" or "rgb_array".

    ## Version History

    * v1: Simplify the max equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, hPolicy = None, render_mode: Optional[str] = None):
        """
        Initializes the RASPendulumEnv.

        Args:
            hPolicy: the critic policy used for training the V policy. If `None`, the environment is in H policy mode.
            render_mode: the rendering mode for the environment. Can be "human" or "rgb_array".
        """

        # Set cart parameters
        self.max_speed = 5.0
        self.max_acceleration = 3.0

        # Set time step
        self.dt = 0.1

        # Checking if we are training the actor or critic
        self.critic = hPolicy == None
        
        # Adding the critic policy
        with torch.device('cpu'):
            self.hPolicy = hPolicy

        # Setting critPolicy to evaluation mode
        if not self.critic:
            self.hPolicy.critic.eval()

        # Defining target
        self.target = 0            # Where is the target?
        self.target_radius = 0.5    # How big is the target?

        # Defining the obstacle
        self.obstacle = -4.8        # Where is the obstacle?
        self.obstacle_radius = 0.2  # How big is the obstacle?

        # Calculating obstacle cost buff, this ensures that the obstacle cost will be visible to the agent instead
        #  of being hidden by the reach cost.
        self.obstacle_buff = -2*min(self.g_x(self.obstacle+self.obstacle_radius), self.g_x(self.obstacle-self.obstacle_radius))/self.obstacle_radius

        # Setting render mode
        self.render_mode = render_mode

        # Set the screen dimensions and initialize the screen and clock placeholders
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        # Create the observation space shape
        high = np.array([abs(self.obstacle) + 2, self.max_speed], dtype=np.float32)

        # Initialize the observation space
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Initialize the action space
        self.action_space = spaces.Box(low=-self.max_acceleration, high=self.max_acceleration, shape=(1,), dtype=np.float32)
    

    def g_x(self, x):
        """
        Calculates the reach cost/reward for the given position.

        Args:
            x: the position of the cart.

        Returns:
            the reach cost/reward for the given position.
        """
        return (self.target_radius - np.sqrt((x - self.target) ** 2))
    
    def l_x(self, x):
        """
        Calculates the avoid cost/reward for the given position.

        Args:
            x: the position of the cart.

        Returns:
            the avoid cost/reward for the given position.
        """
        return (np.sqrt((x - self.obstacle)**2) - self.obstacle_radius) * self.obstacle_buff

    def step(self, u):

        # Unpack the current state
        x, xdot = self.state

        # Clip and extract the control input. In theory this should not be necessary, but it is a good practice.
        u = np.clip(u, -self.max_acceleration, self.max_acceleration)[0]

        # Save the previous control input for rendering
        self.last_u = u

        # Initialize the hFunction reward
        # If we are training V, 
        hReward = None

        # Spinning up a small Batch for the critic value network
        if not self.critic:
            tmp_obs = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
            # Evaluate the critic network
            hReward = self.hPolicy.critic(tmp_obs, torch.tensor([[u]]))

        # Reach: g(x)
        reach = self.g_x(x) if self.critic else hReward.item() if hReward.item() <= 0 else self.g_x(x)
        
        # Avoid: l(x)
        avoid = self.l_x(x)
                 
        # Report worst case reward
        reward = min(reach, avoid)

        newxdot = xdot + u * self.dt
        #newxdot = np.clip(newxdot, -self.max_speed, self.max_speed)
        newx = x + newxdot * self.dt

        self.state = np.array([newx, newxdot])

        # Check if we have left the bounds
        if np.abs(newx) > abs(self.obstacle) + 2:

            # We have left the bounds, report end of episode
            return self._get_obs(), reward, True, False, {"reach":reach,"avoid":avoid}

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, False, False, {"reach":reach,"avoid":avoid}

    def reset(self, is_bounds = True, x_init = None, x_dot_init = None,*, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_XDOT])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = x_init if not None else DEFAULT_X
            y = x_dot_init if not None else DEFAULT_XDOT
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])
        if is_bounds:
            low = -high  # We enforce symmetric limits.
            self.state = self.np_random.uniform(low=low, high=high)
            self.last_u = None
        else:
            self.state = np.array([x, y])

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        x, xdot = self.state
        return np.array([x, xdot], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render mexod wixout specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        # drawing target

        # Setting target_height and target_widx
        target_height = .5*scale
        target_widx = self.target_radius * scale

        # Calculating location of target
        l, r, t, b = self.target- target_widx / 2, self.target + target_widx / 2, target_height/2, -target_height / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        # Drawing pollygon
        gfxdraw.aapolygon(self.surf, coords, (0, 255, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 255, 0))

        # Setting obstacle_height and obstacle_widx
        obstacle_height = .5*scale
        obstacle_widx = self.obstacle_radius * scale

        # Calculating location of obstacle
        l, r, t, b = self.obstacle - obstacle_widx / 2, self.obstacle + obstacle_widx / 2, obstacle_height/2, -obstacle_height / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        # Drawing pollygon
        gfxdraw.aapolygon(self.surf, coords, (255, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (255, 0, 0))

        # drawing cart
        cart_height = 0.8 * scale
        cart_widx = min(self.target_radius,self.obstacle_radius) * .5 * scale

        # Calculating location of cart
        l, r, t, b = self.state[0]-cart_widx / 2, self.state[0] + cart_widx / 2, cart_height / 2, -cart_height / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        gfxdraw.aapolygon(self.surf, coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, coords, (204, 77, 77))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False