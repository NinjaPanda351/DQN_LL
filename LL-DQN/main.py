# import standard libraries
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from networks.q_network import QNetwork

# import custom modules
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
import utils.config as config

# make sure path is correct
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'networks')))
"""
Run a reinforcement learning agent in the specified environment.

Parameters:
    agent: The DQN or Double DQN agent.
    env_name: Name of the Gym environment.
    n_episodes: Total number of episodes for training.
    max_timesteps: Max steps per episode.
    eps_start: Initial epsilon for epsilon-greedy policy.
    eps_end: Minimum epsilon value.
    eps_decay: Decay factor per episode for epsilon.

Returns:
    scores: Total reward per episode.
    returns: Discounted return G per episode.
    successes: Binary success indicator per episode.
"""

def run_agent(agent, env_name, n_episodes=1000, max_timesteps=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    env = gym.make(env_name)
    env.reset(seed=config.SEED)

    scores, returns, successes = [], [], []
    scores_window = deque(maxlen=100) # running average of scores
    epsilon = eps_start

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=config.SEED)
        total_reward = 0 # sum of all rewards in the episode
        success = 0      # calculate return G
        rewards = []     # flag successful landing

        for t in range(max_timesteps):
            action = agent.select_action(state, epsilon) # episol-greedy action selection
            next_state, reward, done, truncated, _ = env.step(action) # environment setup
            agent.step(state, action, reward, next_state, done or truncated) # agent learning
            state = next_state
            total_reward += reward
            rewards.append(reward)
            if done or truncated:
                break
        
        # calculate return g (discounted return)
        G = 0
        gamma = config.GAMMA
        for t, r in enumerate(rewards):
            G += (gamma ** t) * r

        # determine if lander is successefull
        success = 1 if total_reward >= 200 else 0

        # track metrics
        scores.append(total_reward)
        returns.append(total_reward)
        successes.append(success)
        scores_window.append(total_reward)
        epsilon = max(eps_end, eps_decay * epsilon)

        # print episode summmary
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Return G: {G:.2f} | Success: {success}")

    return scores, returns, successes

if __name__ == "__main__":
    env_name = config.ENV_NAME
    
    # train vanilla DQN agent
    print("\nTraining DQN Agent...")
    dqn_agent = DQNAgent(state_dim=8, action_dim=4, seed=config.SEED)
    scores_dqn, returns_dqn, successes_dqn = run_agent(dqn_agent, env_name)

    # train double DQN agent
    print("\nTraining Double DQN Agent...")
    double_dqn_agent = DoubleDQNAgent(state_dim=8, action_dim=4, seed=config.SEED)
    scores_ddqn, returns_ddqn, successes_ddqn = run_agent(double_dqn_agent, env_name)

    # plot learning curve
    episodes = np.arange(len(scores_dqn))
    plt.figure(figsize=(15, 10))

    # plot episodic reward
    plt.subplot(3, 1, 1)
    plt.plot(episodes, scores_dqn, label="DQN")
    plt.plot(episodes, scores_ddqn, label="Double DQN")
    plt.ylabel("Episodic Reward")
    plt.legend()

    # plot return G
    plt.subplot(3, 1, 2)
    plt.plot(episodes, returns_dqn, label="DQN")
    plt.plot(episodes, returns_ddqn, label="Double DQN")
    plt.ylabel("Return G")
    plt.legend()

    # plot success rate
    plt.subplot(3, 1, 3)
    plt.plot(episodes, successes_dqn, label="DQN")
    plt.plot(episodes, successes_ddqn, label="Double DQN")
    plt.ylabel("Success (1/0)")
    plt.xlabel("Episode")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # print summary table of last 100 episodes
    data = {
        "Metric": ["Avg Episodic Reward", "Avg Return", "Success Rate (%)"],
        "DQN (Vanilla)": [
            f"{np.mean(scores_dqn[-100:]):.2f}",
            f"{np.mean(returns_dqn[-100:]):.2f}",
            f"{100 * np.mean(successes_dqn[-100:]):.2f}%"
        ],
        "DQN + Extension": [
            f"{np.mean(scores_ddqn[-100:]):.2f}",
            f"{np.mean(returns_ddqn[-100:]):.2f}",
            f"{100 * np.mean(successes_ddqn[-100:]):.2f}%"
        ]
    }

    df = pd.DataFrame(data)
    print("\nSummary Table (Last 100 Episodes):")
    print(df.to_string(index=False))
