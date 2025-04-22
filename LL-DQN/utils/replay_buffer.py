import random
import numpy as np
import torch
from collections import deque, namedtuple

import utils.config as config


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples for training the DQN.
    """

    def __init__(self, action_dim: int, buffer_capacity: int, batch_size: int, seed: int):
        """
        Initialize a ReplayBuffer object.

        Args:
            action_dim (int): Number of possible actions (for compatibility)
            buffer_capacity (int): Maximum number of experiences to store
            batch_size (int): Number of experiences to sample for training
            seed (int): Random seed for reproducibility
        """
        self.action_dim = action_dim
        self.memory = deque(maxlen=buffer_capacity)  # FIFO buffer
        self.batch_size = batch_size
        self.seed = random.seed(seed)

        # Define a named tuple to store a single experience
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"]
        )

    def add(self, state, action, reward, next_state, done) -> None:
        """
        Add a new experience to memory.
        """
        new_experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(new_experience)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.

        Returns:
            Tuple[torch.Tensor]: Batches of states, actions, rewards, next_states, and done flags.
        """
        sampled_experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([exp.state for exp in sampled_experiences])
        ).float().to(config.DEVICE)

        actions = torch.from_numpy(
            np.vstack([exp.action for exp in sampled_experiences])
        ).long().to(config.DEVICE)

        rewards = torch.from_numpy(
            np.vstack([exp.reward for exp in sampled_experiences])
        ).float().to(config.DEVICE)

        next_states = torch.from_numpy(
            np.vstack([exp.next_state for exp in sampled_experiences])
        ).float().to(config.DEVICE)

        dones = torch.from_numpy(
            np.vstack([exp.done for exp in sampled_experiences]).astype(np.uint8)
        ).float().to(config.DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Return the current size of internal memory.
        """
        return len(self.memory)
