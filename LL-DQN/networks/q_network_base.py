from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Standard Q-Network for base DQN."""

    def __init__(self, state_dim: int, action_dim: int, seed: int) -> None:
        """
        Initialize the Q-network.

        Args:
            state_dim: Dimension of state space.
            action_dim: Number of actions.
            seed: Random seed for reproducibility.
        """
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)
        self.network: nn.Sequential = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for each action.

        Args:
            state: State input tensor of shape (batch_size, state_dim).

        Returns:
            Q-values tensor of shape (batch_size, action_dim).
        """
        return self.network(state)