import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import namedtuple

import gymnasium as gym

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

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
    def __init__(self, env:gym.Env, n_actor_layers, n_critic_layers):
        # initialize environment and exrtract state and action space
        self.env = env
        self.n_actions = self.env.action_space.n
        state,info = self.env.reset()
        self.n_observations = len(state)

        # initialize actor and critic functions
        self.actor = Actor(self.n_observations,self.n_actions,n_actor_layers)
        self.critic = Critic(self.n_observations,self.n_actions,n_critic_layers)

        # initialize buffers to safe rewards during episodes
        self.actions = []
        self.rewards = []
    
    def __select_action__(self, s):
        s = torch.from_numpy(s).float()

        # get action probabilitys and value estimates
        probs = self.actor(s)
        value = self.critic(s)
        a_dist = Categorical(probs)

        # sample an action using probability estimates
        a = a_dist.sample()

        # safe action probab and value to buffer
        self.actions.append(SavedAction(a_dist.log_prob(a),value))

        return a.item()

    def __update_actor__(self, loss):
        pass
    
    def __update_critic__(self, loss):
        pass

    def __get_delta__(self):
        pass

    def optimize(self, budget, gamma):
        # sample episodes within a given budget
        while budget>0:
            s, _  = self.env.reset()
            terminated = False
            epsidoe_return = 0

            # sample single episode TODO: sample more than one using vecorized environments and just use 2D buffers
            while not terminated:
                # select action based on actor
                a = self.__select_action__(s)

                # take action in environment
                s, r, terminated, truncated, _ = self.env.step(a)
                
                # append reward to buffer
                self.rewards.append(r)
                
                terminated = terminated or truncated

                budget -=1
            #TODO Calculate loss and update actor and critic
        
