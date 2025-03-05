# Gabriel Chenevert
# 2/3/2025

import argparse
import os
import pprint
import gymnasium as gym
import customENV.customENVInit
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.exploration import GaussianNoise
from policy.RASddpgZeroSumH import ras_DDPGZeroSum as DDPGPolicy
from tianshou.trainer import OffpolicyTrainer as offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

# Defining the command line arguments for gym
def get_args():
    parser = argparse.ArgumentParser()

    """ Gym Environment Parameters"""
    parser.add_argument('--task', type=str, default='Combine-v1')   # Set the task to be the Fly environment
    
    # Environment termination perameters
    parser.add_argument('-e','--do-term', action="store_true", default=False)  # Should the simulation end when the drone exits the environment?
    parser.add_argument('--term-reward', type=float, default=None)  # Reward for exiting the environment
    
    """ Tianshou DDPG Training Parameters"""
    # Set the default reward threshold for the gymnasium environment, the simulation will end when the rewards hits this threshold
    # Note: this includes both the reward value AND its error bar (i.e. a reward of 800 +- 200 will end the simulation at 1000)
    parser.add_argument('--reward-threshold', type=float, default=950)
    parser.add_argument('-t','--do-threshold', action="store_true", default=False)  # Should the simulation end when the reward threshold is met?
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=40000)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)

    # Device settings
    # Use CUDA if we have it
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # Number of training envorinments (training workers)
    parser.add_argument('--training-num', type=int, default=16)
    # Number of testing environments (testing workers)
    parser.add_argument('--test-num', type=int, default=10)

    """ Training Length and Batch Size Parameters"""
    # Set the number of epochs for the simulation. Note: if --reward-threshold is met, the simulation will end early
    # If reward threshold is not met, the simulation may end with an AssertionError on line 146 (assert stop_fn(result['best_reward'])
    parser.add_argument('--epoch', type=int, default=200)
    # Set the number of steps per epoch. The total steps can be calculated by multiplyaing --step-per-epoch by --epoch
    parser.add_argument('--step-per-epoch', type=int, default=40000)
    parser.add_argument('--step-per-collect', type=int, default=16)
    parser.add_argument('--update-per-step', type=float, default=0.125)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--actor-hidden-sizes', type=int, nargs='*', default=[1024]*4)
    parser.add_argument('--critic-hidden-sizes', type=int, nargs='*', default=[768]*4)

    """ Logging and Rendering Parameters"""
    parser.add_argument('--logdir', type=str, default='log')                # Set the log directory
    parser.add_argument('--doRender', action="store_true", default=False)   # Should the simulation be rendered?
    parser.add_argument('--render', type=float, default=1/30)               # Set the rendering speed
    parser.add_argument('--render-type', type=str, default='human')         # Set the rendering type
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=1)
    args = parser.parse_known_args()[0]
    return args

# Defining ddpg test function
# This is a generic catch all function that *should* work for all systems
def H_ddpg(args=get_args(), task = None, epoch = None, do_term = None, term_reward = None, logdir = None, h_batch_size = None, hidden_sizes = None, steps_per_epoch = None, critic_hidden_sizes = None, **kwargs):

    # Parse the arguments received from external function call
    task = args.task if task is None else task                              # Set new task if provided
    epoch = args.epoch if epoch is None else epoch                          # Set new number of epochs if provided
    do_term = args.do_term if do_term is None else do_term                  # Set new termination parameters if provided
    term_reward = args.term_reward if term_reward is None else term_reward  # Set new termination reward if provided
    logdir = args.logdir if logdir is None else logdir                      # Set new log directory if provided
    batch_size = args.batch_size if h_batch_size is None else h_batch_size      # Set new batch size if provided
    actor_hidden_sizes = args.actor_hidden_sizes if hidden_sizes is None else hidden_sizes  # Set new hidden sizes if provided
    critic_hidden_sizes = args.critic_hidden_sizes if critic_hidden_sizes is None else critic_hidden_sizes  # Set new hidden sizes if provided
    steps_per_epoch = args.step_per_epoch if steps_per_epoch is None else steps_per_epoch  # Set new steps per epoch if provided

    # Print h_batch_size
    print("batch_size: ", batch_size)

    env = gym.make(task)
    args.state_shape = env.observation_space.shape or env.observation_space.n

    # Set up control shape, action
    args.control_shape = env.control_space.shape or env.control_space.n
    args.max_control = env.control_space.high[0]

    # Set up disturbance shape, action
    args.disturbance_shape = env.disturbance_space.shape or env.disturbance_space.n
    args.max_disturbance = env.disturbance_space.high[0]

    # Set up action shape
    args.action_shape = env.action_space.shape or env.action_space.n

    if args.reward_threshold is None:
        default_reward_threshold = {"Combine-v1": 500}
        args.reward_threshold = default_reward_threshold.get(
            task, env.spec.reward_threshold
        )
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(task)
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(task, do_term = do_term, term_reward = term_reward)  for _ in range(args.training_num)],
        context = 'spawn',share_memory=False
    )
    # test_envs = gym.make(task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(task, do_term = do_term, term_reward = term_reward)  for _ in range(args.training_num)],
        context = 'spawn',share_memory=False
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    
   # Control model
    net = Net(args.state_shape, hidden_sizes=actor_hidden_sizes, device=args.device, activation=torch.nn.LeakyReLU)
    control = Actor(
        net, args.control_shape, max_action=args.max_control, device=args.device
    ).to(args.device)
    control_optim = torch.optim.Adam(control.parameters(), lr=args.actor_lr)
    
    # Disturbance model
    net = Net(args.state_shape, hidden_sizes=actor_hidden_sizes, device=args.device, activation=torch.nn.LeakyReLU)
    disturbance = Actor(
        net, args.disturbance_shape, max_action=args.max_disturbance, device=args.device
    ).to(args.device)
    disturbance_optim = torch.optim.Adam(disturbance.parameters(), lr=args.actor_lr)
    
    # Setting up Critic
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=critic_hidden_sizes,
        concat=True,
        device=args.device,
        activation=torch.nn.LeakyReLU
    )
    critic = Critic(net, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    policy = DDPGPolicy(
        control = control,
        control_optim = control_optim,
        disturbance = disturbance,
        disturbance_optim = disturbance_optim,
        critic = critic,
        critic_optim = critic_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        estimation_step=args.n_step,
        action_space=env.action_space,
    )
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(logdir, task, 'ddpg')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        # Saving policy and class so we can load it without knowing anything later
        torch.save(policy, os.path.join(log_path, 'hPolicy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold if args.do_threshold else False

    # trainer + Run
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=steps_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=batch_size,
        update_per_step=args.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
    ).run()

    if __name__ == '__main__':
        pprint.pprint(result)


if __name__ == '__main__':
    H_ddpg()