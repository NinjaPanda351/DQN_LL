import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Neural network for approximating Q-values in a DQN agent.
    Maps state observations to Q-values for each possible action.
    """

    def __init__(self, state_dim: int, action_dim: int, seed: int) -> None:
        """
        Initialize the Q-network.

        Args:
            state_dim (int): Dimension of the input state space.
            action_dim (int): Number of possible actions (output Q-values).
            seed (int): Random seed for reproducibility.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)  # Set manual seed for consistent initialization

        # Input layer: maps state to first hidden layer
        self.input_to_hidden = nn.Linear(state_dim, 64)
        # Hidden layer: maps from first to second hidden layer
        self.hidden_to_hidden = nn.Linear(64, 64)
        # Output layer: maps from hidden state to Q-values for each action
        self.hidden_to_output = nn.Linear(64, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): The current environment state.

        Returns:
            torch.Tensor: Predicted Q-values for each action.
        """
        hidden1 = F.relu(self.input_to_hidden(state))  # Apply ReLU after first layer
        hidden2 = F.relu(self.hidden_to_hidden(hidden1))  # Apply ReLU after second layer
        q_values = self.hidden_to_output(hidden2)  # Output layer gives Q-values
        return q_values
