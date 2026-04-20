from tqdm import tqdm
import numpy as np
import torch
import gymnasium as gym
from REINFORCE import PolicyNet
from REINFORCE import train_REINFORCE
import pandas as pd
from tqdm import tqdm

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


n_observations = 4
n_actions = 2
net_length = 1
lr = 3e-4

file_path = f'test_REINFORCE.csv'
print(f'Running experiment: {file_path}')

results = []
for i in range(5):
    curr_env = gym.make("CartPole-v1")
    policy_net = PolicyNet(n_observations, n_actions, net_length).to(device)
    curr_eval_timesteps, curr_eval_returns = train_REINFORCE(
        env=curr_env,
        policy_net=policy_net,
        device=device,
        lr=lr,
        budget = 1e4,
        verbose=0,
        eval_rate=250
    )
    results.append(np.array(curr_eval_returns))
results = np.array(results)
df = pd.DataFrame({
    "eval_timesteps": curr_eval_timesteps,
    "eval_mean_returns":np.mean(results, axis=0),
    "eval_std_returns":np.std(results, axis=0)
    })
df.to_csv(file_path)