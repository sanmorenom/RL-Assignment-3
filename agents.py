import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
from collections import namedtuple
from itertools import count
from time import time

import gymnasium as gym

class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, n_layers):
        super(Actor, self).__init__()
        self.n_layers = n_layers
        self.a_in = nn.Linear(n_observations, 128)
        for idx in range(self.n_layers):
            setattr(self, f'layer{idx+1}', nn.Linear(128,128))
        self.a_out = nn.Linear(128, n_actions)

    def forward(self, observations):
        x = F.relu(self.a_in(observations))
        for idx in range(self.n_layers):
            x = F.relu(getattr(self, 'layer%d' % (idx+1))(x))
        return F.softmax(self.a_out(x), dim=-1)
    
class Critic(nn.Module):
    def __init__(self, n_observations, n_actions, n_layers):
        super(Critic, self).__init__()
        self.n_layers = n_layers
        self.c_in = nn.Linear(n_observations, 128)
        for idx in range(self.n_layers):
            setattr(self, f'layer{idx+1}', nn.Linear(128,128))
        self.c_out = nn.Linear(128, n_actions)

    def forward(self, observations):
        x = F.relu(self.c_in(observations))
        for idx in range(self.n_layers):
            x = F.relu(getattr(self, 'layer%d' % (idx+1))(x))
        return self.c_out(x)
    
class ModelFreeLearner():
    """
    ModelFreeLearner serves as a perent class for the REINFORCE, ActorCritic (AC) and
    Advantage Actor Critic (A2C). 
    This design choice prevents code duplication since the core functionalities in the 
    optimization/ training step is the same for each learner. We initialize the 
    """
    def __init__(self, env:gym.Env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr):
        # initialize environment and exrtract state and action space
        self.env = env
        self.n_actions = self.env.action_space.n
        state,info = self.env.reset()
        self.n_observations = len(state)

        # initialize hyperparameters
        self.gamma = gamma

        # initialize actor and critic functions
        self.actor = Actor(self.n_observations,self.n_actions,n_actor_layers)
        self.critic = Critic(self.n_observations,self.n_actions,n_critic_layers)

        # initialize buffers to safe rewards during episodes
        self.values = []
        self.log_probs = []
        self.rewards = []

        # initilize optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def __select_action__(self, state):
        state = torch.from_numpy(state).float()
        
        # get action probabilitys and  estimates
        probs = self.actor(state)
        action_dist = Categorical(probs)
    
        # sample an action using probability estimates
        action = action_dist.sample()

        # also return log probability since we need it for actor updates
        return action.item(), action_dist.log_prob(action)
    
    def __reset_buffers__(self):
        del self.values[:]
        del self.log_probs[:]
        del self.rewards[:]

    def __update_actor__(self, returns):
        pass

    def __update_critic__(self,returns):
        # loss between target value (calculated from returns) and predicted q_vals
        loss = F.mse_loss(torch.stack(self.values), returns.detach())

        # do gradient decent step
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def __get_returns__(self):
        returns = []
        R = 0 

        # iterate over reversed rewards array 
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0,R)
        return torch.as_tensor(returns, dtype=torch.float32)
    
    def optimize(self, budget):
        iterations = budget
        t0 = time()
        episode_rewards = []
        # sample episodes within a given budget
        while budget>0:
            self.__reset_buffers__()
            state, _  = self.env.reset()

            # sample action from actor (calculate the log prob as well to prevent overhead)
            action, log_prob = self.__select_action__(state)
            
            terminated = False
            episode_reward = 0.0
            # sample single episode 
            for step in count(1):
                # take action in env
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # save info to buffers
                self.values.append(self.critic(torch.tensor(state))[action])
                self.rewards.append(reward)
                self.log_probs.append(log_prob)

                next_action, log_prob = self.__select_action__(state)
                
                # advance state and action
                state = next_state
                action = next_action

                episode_reward += reward 
                budget -=1

                if terminated or truncated:
                    break

            # calculate returns based on rewards 
            returns = self.__get_returns__()

            # update both actor and critic
            self.__update_actor__(returns)
            self.__update_critic__(returns)

            # empty the buffers
            self.__reset_buffers__()

            episode_rewards.append((episode_reward,iterations-budget))
            progress = (((iterations-budget)/iterations)*100)
            eta = (time()-t0)*((100-progress)/progress)
            print(f"\rProgress: {progress:.2f}% ETA: {(eta):.0f}s", end='', flush=True)
        print() 
        return episode_rewards
    


    
class REINFORCE(ModelFreeLearner):
    def __init__(self, env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr):
        super().__init__(env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr)
    def __update_actor__(self, returns):
        # Calculate loss; the - is introduced since we want to acent gradients -> negative loss gradient descent 
        
        loss = torch.sum(-torch.stack(self.log_probs) * returns.detach())

        # do gradient decent step
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    # override functions to prevent computational overhead   
    def __update_critic__(self, returns):
        pass
    def __get_deltas__(self, state, action, reward, next_state, next_action):
        pass

class AC(ModelFreeLearner):
    def __init__(self, env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr):
        super().__init__(env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr)
    
    def __update_actor__(self, returns):
        # Calculate loss; the - is introduced since we want to acent gradients -> negative loss gradient descent 
        loss = torch.sum(-torch.stack(self.log_probs) * torch.stack(self.values).detach())

        # do gradient decent step
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()


class A2C(ModelFreeLearner):
    def __init__(self, env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr):
        super().__init__(env, n_actor_layers, n_critic_layers, gamma, actor_lr, critic_lr)
    
    def __update_actor__(self, returns):
        # calculate advantages
        advantages = returns - torch.stack(self.values).detach()
        # consider normalizing advantages for stability 
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 
        loss = torch.sum(-torch.stack(self.log_probs) * advantages)

        # do gradient decent step
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()



if __name__ == "__main__":

    # quick test run; The episode returns are maximised at roughly 98 since we are using a discount factor 
    env = gym.make("CartPole-v1")
    actor_critic = AC(env,1,1,0.99, 0.001,0.01)
    actor_critic.optimize(200000)
