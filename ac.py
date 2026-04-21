import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import namedtuple
from itertools import count

import gymnasium as gym

class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, n_layers):
        super(Actor, self).__init__()
        self.n_layers = n_layers
        self.a_in = nn.Linear(n_observations, 128)
        for idx in range(self.n_layers):
            setattr(self, f'layer_{idx+1}', nn.Linear(128,128))
        self.a_out = nn.Linear(128, n_actions)

    def forward(self, observations):
        x = F.relu(self.a_in(observations))
        for idx in range(self.n_layers):
            x = F.relu(getattr(self, 'layer%d' % (idx+1))(x))
        return F.softmax(self.a_out(x))
    
class Critic(nn.Module):
    def __init__(self, n_observations, n_actions, n_layers):
        super(Critic, self).__init__()
        self.n_layers = n_layers
        self.c_in = nn.Linear(n_observations, 128)
        for idx in range(self.n_layers):
            setattr(self, f'layer_{idx+1}', nn.Linear(128,128))
        self.c_out = nn.Linear(128, n_actions)

    def forward(self, observations):
        x = F.relu(self.c_in(observations))
        for idx in range(self.n_layers):
            x = F.relu(getattr(self, 'layer%d' % (idx+1))(x))
        return self.c_out(x)
    
class ActorCritic():
    """
    Actor Crtitc implementation
    """
    def __init__(self, env:gym.Env, n_actor_layers, n_critic_layers, gamma):
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
        self.states = []
        self.actions = []
        self.rewards = []
        self.deltas = []
    
    def __select_action__(self, s):
        s = torch.from_numpy(s).float()
        # get action probabilitys and  estimates
        probs = self.actor(s)
        a_dist = Categorical(probs)
        # sample an action using probability estimates
        a = a_dist.sample()

        return a.item()
    
    def __reset_buffers__(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.deltas[:]

    def __update_actor__(self):
        # update critic by calculating log probs for each state action pair
        pass
    def __update_critic__(self, loss):
        pass

    def __get_delta__(self,state, action, reward, next_state, next_action):
        q_value = self.critic(state)[:,action]
        next_q_value = self.critic(next_state)[:,next_action]
        delta = (reward + self.gamma * q_value) - next_q_value
        return delta

    def optimize(self, budget, gamma):
        episode_rewards = []
        # sample episodes within a given budget
        while budget>0:
            self.__reset_buffers__()
            state, _  = self.env.reset()
            
            terminated = False
            action = self.__select_action__(state)
            episode_reward = 0.0
            # sample single episode TODO: sample more than one using vecorized environments and just use 2D buffers
            for step in count(1):
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_action = self.__select_action__(next_state) 
                delta = self.__get_delta__(state,action,reward,next_state,next_action)

                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.deltas.append(delta)

                state = next_state
                action = next_action

                episode_reward += reward * (self.gamma ** step)
                budget -=1
                if terminated or truncated:
                    break
            self.__update_actor__()
            self.__update_critic__()
            episode_rewards.append(episode_reward)
        return episode_rewards
