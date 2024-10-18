import torch
import numpy as np

class Actor():
    """
    Represents an actor for the RA DDPG systems
    """

    def __init__(self, device, ra):
        """
        Initializes the Actor object.

        Loads the RA network from the specified file path and sets it to evaluation mode.

        Args:
            device (device): The device to run the models on.
            ra (network): The RA function network.
        """

        # Set the device
        self.device = device

        # Load the RA network
        self.RA = ra

        # Set the RA network to evaluation mode
        self.RA.eval()

    def getResult(self, state):
        """
        Gets the result of the actor based on the given state.

        Args:
            state : The state of the RA DDPG system. As an array of 32 bit floats

        Returns:
            float: The action determined by the actor.
        """

        # Get the control from the RA network
        u = self.RA.control(state)

        # Get the disturbance from the RA network
        d = self.RA.disturbance(state)

        # Concatenate the control and disturbance
        u = torch.cat((u[0], d[0]), 1)

        # Get RA critic reward
        raReward = self.RA.critic(state, u).item()

        return u[0], raReward