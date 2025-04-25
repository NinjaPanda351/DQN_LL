import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from agents.dqn_agent import DQNAgent
import utils.config as config
import gymnasium as gym

# Initialize environment and agent
env = gym.make(config.ENV_NAME)
env.reset(seed=config.SEED)
agent = DQNAgent(state_dim=8, action_dim=4, seed=config.SEED)


def dqn(n_episodes=1000, max_timesteps=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Deep Q-Learning training loop.

    Returns:
        list of scores per episode
    """
    scores = []                          # scores from each episode
    scores_window = deque(maxlen=100)    # rolling window of the last 100 scores
    epsilon = eps_start

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=config.SEED)
        total_reward = 0

        for t in range(max_timesteps):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, done or truncated)
            state = next_state
            total_reward += reward
            if done or truncated:
                break

        scores_window.append(total_reward)
        scores.append(total_reward)
        epsilon = max(eps_end, eps_decay * epsilon)

        print(f"\rEpisode {episode}\tAverage Score: {np.mean(scores_window):.2f}", end="")
        if episode % 100 == 0:
            print(f"\rEpisode {episode}\tAverage Score: {np.mean(scores_window):.2f}")

        if np.mean(scores_window) >= 200.0:
            print(f"\nEnvironment solved in {episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}")
            torch.save(agent.online_q_network.state_dict(), 'checkpoint.pth')
            break

    return scores


if __name__ == "__main__":
    training_scores = dqn()

    # Plot results
    plt.plot(np.arange(len(training_scores)), training_scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.title("DQN Training Performance")
    plt.show()