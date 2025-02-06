# Gabriel Chenevert
# 2/3/2025

import argparse
from training.RASCombineH import H_ddpg
from training.RASCombineV import V_ddpg
import torch
import numpy as np
import gymnasium as gym
from modelCall.RASZeroSumCall import Actor

DEFAULT_X = -5.0
DEFAULT_Y = -2.0
DEFAULT_THETA = 0.0
DEFAULT_VT = 0.0
DEFAULT_VC = 0.6

def get_args():

    parser = argparse.ArgumentParser(description="Train a RASCombine agent")

    # Task
    parser.add_argument("--task", type=str, default="Combine-v1", help="Task to train on")

    # Number of epochs
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs to train for")

    # Environment termination parameters
    parser.add_argument('-e', "--do-terminate", action='store_true',default=False, help="Whether to terminate the environment when the episode is done")
    parser.add_argument('--term-reward', type=float, default=None, help="Reward to give when the episode terminated early")

    # Logging directory
    parser.add_argument('--logdir', type=str, default='log')

    # Use CUDA if we have it
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Network sized
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[ 128, 128])

    # Testing parameters                                          # Time to test
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

    # Test the agent
    if doTest:
        test(args.state, args.task, args.logdir, args.device, args.csv)

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
        f.write("X, Y, Z, Xdot, Ydot, Zdot, vx, vy, Reward, h, v\n")

    # Close the file
    f.close()

    # Run the environment until we get 20 positive rewards or run for 1000 steps
    while reward_count < 20 and steps < 1000:

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
            f.write(f'{state[0]}, {state[1]}, {state[2]}, {state[3]}, {state[4]}, {reward}, {hReward}, {vReward}\n')
    
    # Close the file
    f.close()

    # Print out the final state
    print("Final State: ", state)

if __name__ == "__main__":
    main(train = True, doTest=True)