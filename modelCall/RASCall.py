import torch
import numpy as np

class Actor():
    """
    Represents an actor for RAS DDPG systems.
    
    Args:
        valuePath (str): The file path to load the Value network.
        hPath (str): The file path to load the H network.
    """
    
    def __init__(self, device, h, v):
        """
        Initializes the Actor object.
        
        Stores the H and V policy networks and sets them to evaluation mode.
        
        Args:
            h (torch network): The H policy network.
            v (torch network): The V policy network.
        """

        # Set the device
        self.device = device

        # Save the Value and H policies
        self.V = v
        self.H = h

        # Set the V policy to evaluation mode
        self.V.eval()

        # Set the H policy to evaluation mode
        self.H.eval()
    
    def getResult(self, state):
        """
        Gets the result of the actor based on the given state.
        
        Args:
            state (numpy array): The state of a RAS DDPG system.
        
        Returns:
            h/v u (numpy array): The action determined by the actor.
            hReward (float): The reward from the H critic.
            vReward (float): The reward from the V critic.
        """

        # Get the control from the H network
        hu = self.H.actor(state)

        # Get H critic reward
        hReward = self.H.critic(state, hu[0]).item()

        # Get the control from the V network
        vu = self.V.actor(state)[0]

        # Get V critic reward
        vReward = self.V.critic(state, vu).item()
        
        # If the critic of the H network is greater than 0, return the action from the H network
        if hReward > 0:

            # Extracting the control output from the tensor, and returning it
            return hu[0], hReward, vReward
        else:
            
            # Extracting the control output from the tensor, and returning it
            return vu, hReward, vReward