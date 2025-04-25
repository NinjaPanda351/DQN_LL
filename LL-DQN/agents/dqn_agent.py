from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

from networks.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer
import utils.config as config


class DQNAgent:
    """Deep Q-Learning Agent with Dueling DQN architecture."""

    def __init__(self, state_dim: int, action_dim: int, seed: int) -> None:
        """
        Initialize the agent.

        Args:
            state_dim: Number of state features.
            action_dim: Number of possible actions.
            seed: Random seed for reproducibility.
        """
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.seed: int = seed
        random.seed(seed)
        torch.manual_seed(seed)

        # Q-Networks (Dueling architecture)
        self.online_q_network: QNetwork = QNetwork(state_dim, action_dim, seed).to(config.DEVICE)
        self.target_q_network: QNetwork = QNetwork(state_dim, action_dim, seed).to(config.DEVICE)
        self.optimizer: optim.Adam = optim.Adam(self.online_q_network.parameters(), lr=config.LR)

        # Replay buffer
        self.replay_buffer: ReplayBuffer = ReplayBuffer(
            action_dim,
            config.BUFFER_SIZE,
            config.BATCH_SIZE,
            seed
        )

        # Step counter for updates
        self.train_step_counter: int = 0

        # Precomputed state normalization for LunarLander-v2
        self.state_mean: np.ndarray = np.array([0.0] * state_dim)
        self.state_std: np.ndarray = np.array([1.5, 1.5, 5.0, 5.0, 3.0, 3.0, 1.0, 1.0])

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using precomputed mean and std."""
        return (state - self.state_mean) / (self.state_std + 1e-6)

    def step(
        self,
        current_state: np.ndarray,
        action_taken: int,
        reward_received: float,
        next_state: np.ndarray,
        episode_done: bool
    ) -> Optional[float]:
        """
        Store experience and train the network.

        Args:
            current_state: Current state.
            action_taken: Action taken.
            reward_received: Reward received.
            next_state: Next state.
            episode_done: Whether episode is done.

        Returns:
            Loss value if training occurred, else None.
        """
        current_state = self.normalize_state(current_state)
        next_state = self.normalize_state(next_state)
        reward_received = np.clip(reward_received, -1.0, 1.0)

        self.replay_buffer.add(current_state, action_taken, reward_received, next_state, episode_done)

        self.train_step_counter = (self.train_step_counter + 1) % config.UPDATE_EVERY
        if self.train_step_counter == 0 and len(self.replay_buffer) > config.BATCH_SIZE:
            experience_batch = self.replay_buffer.sample()
            return self.learn_from_experience(experience_batch, config.GAMMA)
        return None

    def select_action(self, current_state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            current_state: Current state.
            epsilon: Exploration rate.

        Returns:
            Action index.
        """
        current_state = self.normalize_state(current_state)
        state_tensor: torch.Tensor = torch.from_numpy(current_state).float().unsqueeze(0).to(config.DEVICE)

        self.online_q_network.eval()
        with torch.no_grad():
            predicted_q_values: torch.Tensor = self.online_q_network(state_tensor)
        self.online_q_network.train()

        if random.random() > epsilon:
            return int(np.argmax(predicted_q_values.cpu().data.numpy()))
        return random.choice(np.arange(self.action_dim))

    def learn_from_experience(
        self, experiences: Tuple[torch.Tensor, ...], discount_factor: float
    ) -> float:
        """
        Learn from a batch of experiences.

        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones).
            discount_factor: Gamma for reward discounting.

        Returns:
            Loss value for logging.
        """
        states, actions, rewards, next_states, dones = experiences

        # Standard DQN with Dueling architecture
        next_q_values: torch.Tensor = self.target_q_network(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_values: torch.Tensor = rewards + (discount_factor * next_q_values * (1 - dones))
        predicted_q_values: torch.Tensor = self.online_q_network(states).gather(1, actions)

        loss: torch.Tensor = F.mse_loss(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.soft_update_networks(self.online_q_network, self.target_q_network, config.TAU)
        return loss.item()

    def soft_update_networks(self, source_model: QNetwork, target_model: QNetwork, tau: float) -> None:
        """Soft update target network parameters."""
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)