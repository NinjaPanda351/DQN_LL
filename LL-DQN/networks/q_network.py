from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Dueling Q-Network with value and advantage streams."""

    def __init__(self, state_dim: int, action_dim: int, seed: int) -> None:
        """
        Initialize the dueling Q-network.

        Args:
            state_dim: Dimension of state space.
            action_dim: Number of actions.
            seed: Random seed for reproducibility.
        """
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)

        # Shared feature layer
        self.feature_layer: nn.Sequential = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Value stream
        self.value_stream: nn.Sequential = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # V(s)
        )

        # Advantage stream
        self.advantage_stream: nn.Sequential = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)  # A(s, a)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for each action.

        Args:
            state: State input tensor of shape (batch_size, state_dim).

        Returns:
            Q-values tensor of shape (batch_size, action_dim).
        """
        features: torch.Tensor = self.feature_layer(state)
        value: torch.Tensor = self.value_stream(features)
        advantages: torch.Tensor = self.advantage_stream(features)

        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        q_values: torch.Tensor = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values