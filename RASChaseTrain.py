# Gabriel Chenevert
# 2/18/2024

import argparse
from training.RASChaseV import V_ddpg
from training.RASChaseH import H_ddpg
import torch
import numpy as np
import gymnasium as gym
from modelCall.RASZeroSumCall import Actor

# Default environment start position
DEFAULT_X = 0.8
DEFAULT_Y = -0.7
DEFAULT_Z = .1
DEFAULT_XDOT = 0.0
DEFAULT_YDOT = 0.0
DEFAULT_ZDOT = 0.0
DEFAULT_VX = 0.0
DEFAULT_VY = 0.0
DEFAULT_VXDOT = 0.0
DEFAULT_VYDOT = 0.0

def get_args():
    parser = argparse.ArgumentParser()

    # Task
    parser.add_argument('--task', type=str, default='DroneChase-v1')

    # Number of epochs
    parser.add_argument('--epoch', type=int, default=500)

    # Environment termination perameters
    parser.add_argument('-e','--do-term', action="store_true", default=False)  # Should the simulation end when the drone exits the environment?
    parser.add_argument('--term-reward', type=float, default=None)  # Reward for exiting the environment

    # Logging directory
    parser.add_argument('--logdir', type=str, default='log')

    # Use CUDA if we have it
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Network sized
    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[ 800, 800, 800])

    # Testing parameters                                          # Time to test
    parser.add_argument('--state', type=np.array, default=np.array([DEFAULT_X, DEFAULT_Y, DEFAULT_Z, 
                                                                    DEFAULT_XDOT, DEFAULT_YDOT, DEFAULT_ZDOT, DEFAULT_VX, DEFAULT_VY, DEFAULT_VXDOT, DEFAULT_VYDOT])) # State to test
    parser.add_argument('--csv', type=str, default='chaseCheck.csv')                                              # CSV file to store the results
    parser.add_argument('--res', type=float, default=0.05)                                                        # Resolution for checking the value functions
    parser.add_argument('--h-csv', type=str, default='hCheck.csv')                                                # CSV file to store the H value function
    parser.add_argument('--v-csv', type=str, default='vCheck.csv')                                                # CSV file to store the V value function
    parser.add_argument('--r-csv', type=str, default='rCheck.csv')                                                # CSV file to store the reward function

    # Parse the arguments
    args = parser.parse_known_args()[0]

    return args

def train(args=get_args()):

    # Train H function
    H_ddpg(**vars(args))

    # Move h function to chase v2
    torch.save(torch.load(args.logdir + '/' + args.task + '/ddpg/hPolicy.pth'), args.logdir + '/DroneChase-v2/ddpg/hPolicy.pth')

    # Train V function
    V_ddpg(**vars(args))

    # Copy v function to chaes v1
    torch.save(torch.load(args.logdir + '/' + args.task + '/ddpg/vPolicy.pth'), args.logdir + '/DroneChase-v1/ddpg/vPolicy.pth')

    test(args.state, args.task, args.logdir, args.device, args.csv, forceV=True)

    value(args.res, args.task, h = torch.load(args.logdir + '/' + args.task + '/ddpg/hPolicy.pth') , v = torch.load(args.logdir + '/' + args.task + '/ddpg/vPolicy.pth'), h_csv = args.h_csv, v_csv = args.v_csv, r_csv = args.r_csv)

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
            f.write(f'{state[0]}, {state[1]}, {state[2]}, {state[3]}, {state[4]}, {state[5]}, {state[6]}, {state[7]}, {reward}, {hReward}, {vReward}\n')
    
    # Close the file
    f.close()

    # Print out the final state
    print("Final State: ", state)

def value(res, environment, h = None, v = None, h_csv = None, v_csv = None, r_csv = None):

    # Load the environment
    env = gym.make(environment)

    # Create array of x positions in the environment
    x = np.arange(-env.x_bound, env.x_bound, res)

    # Create array of y positions in the environment
    y = np.arange(-env.y_bound, env.y_bound, res)

    # Create array of z positions in the environment
    z = np.arange(0, env.z_bound, res)

    # Set vehicle position to 0,0,0,0
    v_state = np.array([0, 0, 0, 0])

    # Construct the state
    state = np.array([0,0,0,0,0,0,v_state[0],v_state[1],v_state[2],v_state[3]])

    # Check if h function is provided
    if h is not None:

        # Create a csv to store state, control and reward
        with open(h_csv, 'w', newline = '') as f:

            # Write the header
            f.write("X, Y, Z, Reward\n")

            # Iterate through x positions
            for i in x:

                # Iterate through y positions
                for j in y:

                    # Iterate through z positions
                    for k in z:

                        # Set the state
                        state = np.array([i, j, k, 0, 0, 0, 0, 0, 0, 0])

                        # Convert to tensor
                        tensor_state = torch.tensor([state], dtype=torch.float64)

                        # Get the control from the H network
                        temp_u = h.control(tensor_state)

                        # Get the disturbance from the H network
                        temp_d = h.disturbance(tensor_state)

                        # Concatenate the control and disturbance
                        temp_u = torch.cat((temp_u[0], temp_d[0]), 1)

                        # Get the reward from the H network
                        reward = h.critic(tensor_state, temp_u).item()

                        # Write the state and reward to the csv
                        f.write(f'{state[0]}, {state[1]}, {state[2]}, {reward}\n')
    
        # Close the file
        f.close()

    # Check if v function is provided
    if v is not None:

        # Create a csv to store state, control and reward
        with open(v_csv, 'w', newline = '') as f:

            # Write the header
            f.write("X, Y, Z, Reward\n")

            # Iterate through x positions
            for i in x:

                # Iterate through y positions
                for j in y:

                    # Iterate through z positions
                    for k in z:

                        # Set the state
                        state = np.array([i, j, k, 0, 0, 0, 0, 0, 0, 0])

                        # Convert to tensor
                        tensor_state = torch.tensor([state], dtype=torch.float64)

                        # Get the control from the V network
                        temp_u = v.control(tensor_state)

                        # Get the disturbance from the V network
                        temp_d = v.disturbance(tensor_state)

                        # Concatenate the control and disturbance
                        temp_u = torch.cat((temp_u[0], temp_d[0]), 1)

                        # Get the reward from the V network
                        reward = v.critic(tensor_state, temp_u).item()

                        # Write the state and reward to the csv
                        f.write(f'{state[0]}, {state[1]}, {state[2]}, {reward}\n')
    
        # Close the file
        f.close()

    # Check if r_csv exists
    if r_csv is not None:

        # Set Control
        u = np.array([0,0,0,0,0])

        # Create a csv to store reward values
        with open(r_csv, 'w', newline='') as f:

            # Write the header
            f.write("X, Y, Z, Reward, g, l\n")

            # Iterate through x positions
            for i in x:

                # Iterate through y positions
                for j in y:

                    # Iterate through z positions
                    for k in z:

                        # Set the state
                        state = np.array([i, j, k, 0, 0, 0, 0, 0, 0, 0])

                        # Reset the environment
                        env.reset(state_init = state)

                        # Step environment
                        info = env.step(u)

                        # Get the reward
                        reward = info[1]

                        # Get g
                        g = info[4]["reach"]

                        # Get l
                        l = info[4]["avoid"]

                        # Write the data to the csv
                        f.write(f'{i}, {j}, {k}, {reward}, {g}, {l}\n')

        # Close the file
        f.close()

if __name__ == '__main__':

    train()