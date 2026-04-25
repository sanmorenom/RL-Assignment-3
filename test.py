"""
Test file for visulizing the results of a single run REINFROCE AC and A2C; saves the result to file result.png
"""



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from agents import A2C,AC,REINFORCE
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


def parse_results(episode_rewards: list[tuple]):
    returns = np.array([r for r, _ in episode_rewards])
    steps   = np.array([s for _, s in episode_rewards])
    return returns, steps


def rolling_average(data, window=20):
    return [np.mean(data[max(0, i - window + 1):i + 1]) for i in range(len(data))]


def plot_results(results: dict[str, list[tuple]], window=20, solved_threshold=200):
    """
    results: dict of {"label": episode_rewards list, ...}
    e.g. {"A2C": a2c_rewards, "AC": ac_rewards}
    """
    colors = ["#1D9E75", "#D85A30", "#378ADD", "#BA7517"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for (label, episode_rewards), color in zip(results.items(), colors):
        returns, steps = parse_results(episode_rewards)
        avg = rolling_average(returns, window)

        ax.plot(steps, returns, color=color, alpha=0.2, linewidth=0.8)
        ax.plot(steps, avg, color=color, linewidth=2, label=label)

    ax.axhline(solved_threshold, color="#E24B4A", linewidth=1.5,
               linestyle="--", label=f"solved ({solved_threshold})")

    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title("actor-critic training — CartPole-v1")
    ax.legend()
    ax.grid(alpha=0.2)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()


env = gym.make("CartPole-v1")

actor_critic = AC(env,2,2,0.99, 6.25e-5)
advantage_actor_critic = A2C(env,2,2,0.99, 6.25e-5)
REINFORCE_agent = REINFORCE(env,2,2,0.99, 6.25e-5)

ac_rewards = actor_critic.optimize(200000)
a2c_rewards = advantage_actor_critic.optimize(200000)
REINFORCE_rewards = REINFORCE_agent.optimize(200000)

# --- usage ---
plot_results({"A2C": a2c_rewards, "AC": ac_rewards, "REINFORCE": REINFORCE_rewards})


