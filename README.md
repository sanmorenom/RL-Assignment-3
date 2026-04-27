# RL-Assignment-3
This repository contains the source code for the 3. Assignment for Reinforcement Learning by Jasper Scheel and Santiago Moreno Mercado. The source code to train the agent is contained in the agents.py file.
## Experiments
The experiment script is contained in the experiment.py file, and all the results are stored as `.csv` files in the Full_Run_Results directory. To get the experiment results plot, simply execute the aforementioned script. Since the results files are not being submitted, they can be downloaded [here](https://github.com/sanmorenom/RL-Assignment-3/tree/58d434150ea900a318c84997ff6387b2a5e2c821/Full_Run_Results)
## Training an Agent
To train an agent, please do the following:
1. Import required libraries
```Python
import gymnasium as gym
from agents import *
```
2. Initialize CartPole environment
```Python
env = gym.make("CartPole-v1")
```

3. Initialize agent training instance with environment, network parameters, and hyperparameters
```Python
# ModelFreeLearner is the parent class for REINFORCE, AC, and A2C
agent = ModelFreeLearner(
  env=env,
  n_actor_layers = 2,
  n_critic_layers = 2,
  gamma = 0.99,
  actor_lr = 0.001,
  critic_lr = 0.001
)
```
4. Train the agent
```Python
agent.optimize(budget = 1000000)
```
## Dependencies
For testing and conducting our experiments, we used a conda environment. All its dependencies are given in the `requirements.txt` file.
