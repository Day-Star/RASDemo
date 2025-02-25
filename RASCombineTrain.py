# Gabriel Chenevert
# 2/3/2025

import argparse
import os
from training.RASCombineH import H_ddpg
from training.RASCombineV import V_ddpg
import torch
import numpy as np
import gymnasium as gym
from modelCall.RASZeroSumCall import Actor

DEFAULT_X = 4.0
DEFAULT_Y = -2.0
DEFAULT_THETA = 0.0
DEFAULT_VT = 0.0
DEFAULT_VC = 0.6

def get_args():

    parser = argparse.ArgumentParser(description="Train a RASCombine agent")

    # Task
    parser.add_argument("--task", type=str, default="Combine-v1", help="Task to train on")

    # Number of epochs
    parser.add_argument("--epoch", type=int, default=30, help="Number of epochs to train for")
    parser.add_argument("--steps_per_epoch", type=int, default=40000, help="Number of steps per epoch")

    # Training parameters
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update parameter")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Environment termination parameters
    parser.add_argument('-e', "--do-term", action='store_true',default=False, help="Whether to terminate the environment when the episode is done")
    parser.add_argument('--term-reward', type=float, default=None, help="Reward to give when the episode terminated early")

    # Logging directory
    parser.add_argument('--logdir', type=str, default='log')

    # Use CUDA if we have it
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Network sized
    parser.add_argument('--v-batch-size', type=int, default=512)
    parser.add_argument('--h-batch-size', type=int, default=1024)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[512, 256, 128])
    parser.add_argument('--critic_hidden_sizes', type=int, nargs='*', default=[768, 384, 192])

    # Testing parameters
    parser.add_argument('--state', type=np.array, default=np.array([DEFAULT_X, DEFAULT_Y, DEFAULT_THETA, DEFAULT_VT, DEFAULT_VC]))  # State to test
    parser.add_argument('--csv', type=str, default='combineCheck.csv')                                              # CSV file to store the results
    parser.add_argument('--res', type=float, default=0.05)                                                        # Resolution for checking the value functions
    parser.add_argument('--h-csv', type=str, default='hCheck.csv')                                                # CSV file to store the H value function
    parser.add_argument('--v-csv', type=str, default='vCheck.csv')                                                # CSV file to store the V value function
    parser.add_argument('--r-csv', type=str, default='rCheck.csv')                                                # CSV file to store the reward function

    # Parse the arguments
    args = parser.parse_known_args()[0]

    return args

def main(args=get_args(), train=False, doTest=False):

    # Train the agent
    if train:

        # Train H function
        H_ddpg(**vars(args))

        # Train V function
        V_ddpg(**vars(args))

        # Create the Combine-v1 directory
        os.makedirs(os.path.join(args.logdir, 'Combine-v1', 'ddpg'), exist_ok=True)

        # Move the V function from Combine-v2 to Combine-v1
        v_path = os.path.join(args.logdir, 'Combine-v2', 'ddpg', 'vPolicy.pth')

        # Load the H function
        v = torch.load(v_path)

        # Save the H function to the Combine-v1 directory
        v_path = os.path.join(args.logdir, 'Combine-v1', 'ddpg', 'vPolicy.pth')

        # Save the H function
        torch.save(v, v_path)

    # Test the agent
    if doTest:

        # Run a general test
        test(args.state, args.task, args.logdir, args.device, args.csv)

        # Construct H csv
        h_csv = os.path.join("H", args.csv)

        # Run a test of the H function
        #test(args.state, args.task, args.logdir, args.device, h_csv, forceH=True)


def test(state, environment, logdir, device, csv, forceH = False, forceV = False):

    # Print the test we are running

    # Set up h path
    h_path = logdir + '/' + environment + '/ddpg/hPolicy.pth'

    # Create the H network
    h = torch.load(h_path).to(device)

    # Set up v path
    v_path = logdir + '/' + environment + '/ddpg/vPolicy.pth'

    # Create the V network
    v = torch.load(v_path).to(device)

    # Set up the actor
    actor = Actor(device, h, v)

    # Set h to evaluation mode
    h.eval()

    # Load the environment
    env = gym.make(environment, hPolicy = h)

    # Check if we are forcing H
    if forceH:

        # Re-roll state while initial H is not >=0
        while actor.getResult(torch.tensor([state], dtype=torch.float64))[1] < 0:

            # Roll a new state
            state = np.random.uniform(env.observation_space.low, env.observation_space.high)
    elif forceV:

        # Re-roll state while initial V is not >=0
        while actor.getResult(torch.tensor([state], dtype=torch.float64))[2] < 0:

            # Roll a new state
            state = np.random.uniform(env.observation_space.low, env.observation_space.high)

    # Reset the environment with the starting position
    state = env.reset(state_init = state)[0]

    # Initialize a reward counter and step counter
    reward_count = 0
    steps = 0

    # Create a csv to store state, control and reward
    with open(csv, 'w', newline = '') as f:

        # Write the header
        f.write("X, Y, Theta, VT, VC, thetaDot, at, ac, Reward, h, v\n")

    # Close the file
    f.close()

    # Run the environment until we get 20 positive rewards or run for 1000 steps
    while reward_count < 500 and steps < 1000:

        # Get the action from the actor
        action, hReward, vReward = actor.getResult(torch.tensor([state], dtype=torch.float64))

        # Process action
        action = action.cpu().detach().numpy()
        
        # Take the action
        state, reward = env.step(action)[0:2]

        # Update the reward count
        if reward >= 0:
            reward_count += 1

        # Update the step count
        steps += 1

        # Write the state and reward to the csv
        with open(csv, 'a', newline='') as f:

            # Write the data
            f.write(f'{state[0]}, {state[1]}, {state[2]}, {state[3]}, {state[4]}, {action[0]}, {action[1]}, {action[2]}, {reward}, {hReward}, {vReward}\n')
    
    # Close the file
    f.close()

    # Print out the final state
    print("Final State: ", state)

if __name__ == "__main__":
    main(train = True, doTest=False)