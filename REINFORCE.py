import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import namedtuple, deque
import random
from tqdm import tqdm
import time

class PolicyNet(nn.Module):
    #def __init__(self, n_observations, n_actions, n_layers = 1):
    #    super(PolicyNet, self).__init__()
    #    self.probabilities = []
    #    self.n_layers = n_layers
    #    self.input = nn.Linear(n_observations, 128)
    #    for idx in range(self.n_layers):
    #        setattr(self, 'layer%d' % (idx+1), nn.Linear(128,128))
    #    self.out = nn.Linear(128, n_actions)
#
    #def forward(self, observations):
    #    x = F.relu(self.input(observations))
    #    for idx in range(self.n_layers):
    #        x = F.relu(getattr(self, 'layer%d' % (idx+1))(x)) 
    #    probabilities = F.softmax(self.out(x), dim=1)
    #    return probabilities
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.input = nn.Linear(state_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.input(state))
        action_probs = F.softmax(self.out(x), dim=-1)
        return action_probs
def select_action(state, policy_net, device):
    curr_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    probs = policy_net(curr_state)
    prob_distribution = Categorical(probs)
    action = prob_distribution.sample()
    #policy_net.probabilities.append(prob_distribution.log_prob(action))
    log_prob = prob_distribution.log_prob(action)
    return action,log_prob

#gotta check still
def evla_policy(policy_net, device, eval_iterations = 3):
    episode_returns = []
    eval_env = gym.make("CartPole-v1")
    for _ in range(eval_iterations):
        episode_return = 0
        s,_ = eval_env.reset()
        s = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        terminated = False
        while not terminated:
            #Just do greedy for evaluation
            with torch.no_grad():
                a =  policy_net(s).max(1).indices.view(1, 1)
            sp, r, terminated, truncated, _ = eval_env.step(a.item())
            terminated = terminated or truncated
            episode_return += r
            sp = torch.tensor(sp, dtype=torch.float32, device=device).unsqueeze(0)
            s = sp
        episode_returns.append(episode_return)
    eval_env.close()
    return np.mean(episode_returns)

def optimize_model(policy_net:PolicyNet, gamma:float, optim, rewards, device, log_probs):
    returns = [None] * len(rewards)
    discount_return = 0
    for i in reversed(range(len(rewards))):
        discount_return = rewards[i] + gamma*discount_return
        returns[i] = discount_return

    loss = -torch.sum(torch.stack(log_probs) * torch.tensor(returns, device=device))
    optim.zero_grad()
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optim.step()
    return policy_net, optim


def train_REINFORCE(
        env,
        policy_net, 
        device,
        budget = 1e6,
        gamma = 0.99,
        lr = 3e-4,
        verbose = 1,
        eval_rate = 250
    ):
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    eval_timesteps = []
    eval_returns = []
    counter = 0
    pbar = tqdm(total=budget)
    torch.autograd.set_detect_anomaly(True)
    while counter < budget:
        s, _ = env.reset()
        state = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        terminated = False
        episode_return = []
        episod_log_probs = []
        while counter < budget and not terminated:
            action,log_prob = select_action(state=state, policy_net=policy_net , device=device)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            terminated = terminated or truncated
            episod_log_probs.append(log_prob)
            episode_return.append(reward)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            state = next_state
            counter += 1

            if counter % eval_rate == 0:
                pbar.update(eval_rate)
                eval_timesteps.append(counter)
                eval_return = evla_policy(policy_net, device)
                eval_returns.append(eval_return)
                if verbose == 1:
                    print(f'Episode return: {eval_return}')
         
        policy_net, optimizer = optimize_model(
                    rewards=episode_return,
                    log_probs=episod_log_probs,
                    policy_net=policy_net,
                    gamma=gamma,
                    optim=optimizer,
                    device=device
                )

            
            
    return eval_timesteps, eval_returns
