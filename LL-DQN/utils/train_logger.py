from typing import List, Dict, Any
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from tabulate import tabulate


class TrainLogger:
    """Logger for training metrics and generating plots."""

    def __init__(self, results_dir: str = "results") -> None:
        """
        Initialize the logger.

        Args:
            results_dir: Directory to save plots and logs.
        """
        self.results_dir: str = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.metrics: Dict[str, List[Dict[str, Any]]] = {
            "base_dqn": [],
            "dueling_dqn": []
        }

    def log_episode(
        self,
        agent_type: str,
        episode: int,
        reward: float,
        return_g: float,
        success: bool,
        avg_loss: float
    ) -> None:
        """
        Log metrics for an episode.

        Args:
            agent_type: "base_dqn" or "dueling_dqn".
            episode: Episode number.
            reward: Total episodic reward.
            return_g: Episodic return (discounted sum).
            success: Whether landing was successful.
            avg_loss: Average loss for the episode.
        """
        self.metrics[agent_type].append({
            "episode": episode,
            "reward": reward,
            "return": return_g,
            "success": success,
            "avg_loss": avg_loss
        })

    def save_plots(self) -> None:
        """Generate and save plots for reward, return, and success rate."""
        for metric in ["reward", "return", "success"]:
            plt.figure(figsize=(10, 6))
            for agent_type in self.metrics:
                if not self.metrics[agent_type]:
                    continue
                episodes = [m["episode"] for m in self.metrics[agent_type]]
                values = [m[metric] for m in self.metrics[agent_type]]
                plt.plot(episodes, values, label=agent_type.replace("_", " ").title())
            plt.xlabel("Episode")
            plt.ylabel(metric.title())
            plt.title(f"{metric.title()} vs. Episode Number")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.results_dir, f"{metric}_plot.png"))
            plt.close()

    def save_summary_table(self) -> None:
        """Generate and save a table of average metrics over last 100 episodes."""
        table_data = []
        headers = ["Metric", "DQN (Vanilla)", "Dueling DQN"]
        for metric in ["reward", "return", "success"]:
            row = [metric.title()]
            for agent_type in ["base_dqn", "dueling_dqn"]:
                if not self.metrics[agent_type]:
                    row.append("-")
                    continue
                last_100 = self.metrics[agent_type][-100:]
                avg = np.mean([m[metric] for m in last_100])
                if metric == "success":
                    avg *= 100  # Convert to percentage
                    row.append(f"{avg:.2f}%")
                else:
                    row.append(f"{avg:.2f}")
            table_data.append(row)

        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
        with open(os.path.join(self.results_dir, "summary_table.txt"), "w") as f:
            f.write(table_str)