# RL-Assignment-3
This repository contains the source code for the 3. Assignment for Reinforcement Learning by Jasper Scheel and Santiago Moreno Mercado. The source code to train the agent is contained in the agents.py file.
## Experiments
The experiment script is contained in the experiment.py file, and all the results are stored as .csv files in the Full_Run_Results directory. To get the experiment results plot, simply execute the aforementioned script.
## Training an Agent
To train an agent, please do the following:
1. Import required libraries
```Python
import gymnasium as gym
from agents import REINFORCE, AC, A2C
```
2. Initialize CartPole environment
```Python
env = gym.make("CartPole-v1")
```

3. Initialize agent training instance with environment, network parameters, and hyperparameters
```Python
ModelFreeLearner(
  env=env,
  n_actor_layers = 2,
  n_critic_layers = 2,
  gamma = 0.99,
  actor_lr = 0.001,
  critic_lr = 0.001
)
```
## Dependencies
For testing and conducting our experiments, we used a conda environment. All its dependencies are given in the `requirements.txt` file.
