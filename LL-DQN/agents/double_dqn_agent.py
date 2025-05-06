import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from networks.q_network import QNetwork
from utils.replay_buffer import ReplayBuffer
import utils.config as config


class DoubleDQNAgent:
    """
    Double Deep Q-Learning Agent that interacts with and learns from the environment.
    This implementation extends the standard DQN by using the online network to select actions
    and the target network to evaluate them, which helps reduce overestimation bias.
    """

    def __init__(self, state_dim: int, action_dim: int, seed: int):
        """
        Initialize the agent.

        Args:
            state_dim (int): Number of input features (state space size)
            action_dim (int): Number of possible actions
            seed (int): Random seed for reproducibility
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seed = random.seed(seed)

        # Q-Networks
        self.online_q_network = QNetwork(state_dim, action_dim, seed).to(config.DEVICE)
        self.target_q_network = QNetwork(state_dim, action_dim, seed).to(config.DEVICE)
        self.optimizer = optim.Adam(self.online_q_network.parameters(), lr=config.LR)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            action_dim,
            config.BUFFER_SIZE,
            config.BATCH_SIZE,
            seed
        )

        # Step counter for controlling update frequency
        self.train_step_counter = 0

    def step(self, current_state, action_taken, reward_received, next_state, episode_done):
        """
        Store experience and train the network every few steps.
        """
        self.replay_buffer.add(current_state, action_taken, reward_received, next_state, episode_done)

        self.train_step_counter = (self.train_step_counter + 1) % config.UPDATE_EVERY
        if self.train_step_counter == 0 and len(self.replay_buffer) > config.BATCH_SIZE:
            experience_batch = self.replay_buffer.sample()
            self.learn_from_experience(experience_batch, config.GAMMA)

    def select_action(self, current_state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            current_state (np.ndarray): Current environment state
            epsilon (float): Exploration rate

        Returns:
            int: Action index
        """
        state_tensor = torch.from_numpy(current_state).float().unsqueeze(0).to(config.DEVICE)
        self.online_q_network.eval()
        with torch.no_grad():
            predicted_q_values = self.online_q_network(state_tensor)
        self.online_q_network.train()

        if random.random() > epsilon:
            return np.argmax(predicted_q_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))

    def learn_from_experience(self, experiences, discount_factor: float):
    
        states, actions, rewards, next_states, dones = experiences

        # Double DQN: Use online network to SELECT actions
        self.online_q_network.eval()
        with torch.no_grad():
            online_next_q_values = self.online_q_network(next_states)
        best_actions = online_next_q_values.argmax(dim=1, keepdim=True)
        self.online_q_network.train()
    
        # Use the target network to EVALUATE these actions
        next_q_values = self.target_q_network(next_states).gather(1, best_actions)
    
        # Compute target Q-values
        target_q_values = rewards + (discount_factor * next_q_values * (1 - dones))

            # Get expected Q-values from the online model
        predicted_q_values = self.online_q_network(states).gather(1, actions)

        # Compute loss and update network
        loss = F.mse_loss(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.online_q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft update the target network parameters
        self.soft_update_networks(self.online_q_network, self.target_q_network, config.TAU)        

    def soft_update_networks(self, source_model: QNetwork, target_model: QNetwork, tau: float):
        """
        Soft update target network parameters using source network.

        Args:
            source_model (QNetwork): The model to copy weights from
            target_model (QNetwork): The model to update
            tau (float): Interpolation factor
        """
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )