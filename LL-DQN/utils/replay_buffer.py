from typing import Tuple
import torch
import numpy as np
import random
from collections import deque

import utils.config as config


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_dim: int, buffer_size: int, batch_size: int, seed: int) -> None:
        """
        Initialize a ReplayBuffer.

        Args:
            action_dim: Dimension of action space.
            buffer_size: Maximum size of buffer.
            batch_size: Size of each training batch.
            seed: Random seed.
        """
        self.action_dim: int = action_dim
        self.buffer: deque = deque(maxlen=buffer_size)
        self.batch_size: int = batch_size
        self.rng: random.Random = random.Random(seed)

    def add(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        """Add a new experience to the buffer."""
        experience: Tuple[np.ndarray, int, float, np.ndarray, bool] = (
            state, action, reward, next_state, done
        )
        self.buffer.append(experience)

    def sample(self) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences."""
        experiences: list = self.rng.sample(self.buffer, k=self.batch_size)

        states: torch.Tensor = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(config.DEVICE)
        actions: torch.Tensor = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(config.DEVICE)
        rewards: torch.Tensor = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(config.DEVICE)
        next_states: torch.Tensor = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(config.DEVICE)
        dones: torch.Tensor = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(config.DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)