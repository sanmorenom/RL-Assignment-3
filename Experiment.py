"""
Test file for visulizing the results of a single run REINFROCE AC and A2C; saves the result to file result.png
"""



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from agents import A2C,AC,REINFORCE, ModelFreeLearner
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import csv
from scipy.signal import savgol_filter

def get_full_run_results(file_name,learner_type:ModelFreeLearner,num_repetitions = 5, actor_lr = 1e-4, critic_lr = 1e-3, budget =  200000):    
    # Specify the directory name
    directory_name = 'Full_Run_Results'

    # Create the directory
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    file_path = f'{directory_name}/{file_name}.csv'
    print(f'Running experiment: {file_path}')

    results = []
    for i in range(num_repetitions):
        curr_env = gym.make("CartPole-v1")

        learner = learner_type(curr_env,2,2,0.99, actor_lr,critic_lr)
        curr_eval_timesteps, curr_eval_returns  = learner.optimize_with_eval(budget)
        results.append(np.array(curr_eval_returns))
    
    results = np.array(results)
    df = pd.DataFrame({
        "eval_timesteps": curr_eval_timesteps,
        "eval_mean_returns":np.mean(results, axis=0),
        "eval_std_returns":np.std(results, axis=0)
        })
    df.to_csv(file_path)

def plot_full_runs(solved_threshold=500, num_repetitions = 5):
    """
    results: dict of {"label": episode_rewards list, ...}
    e.g. {"A2C": a2c_rewards, "AC": ac_rewards}
    """
    items = []
    folder = f'Full_Run_Results/'
    files = os.listdir(folder)
    index = 0

    while index < len(files):
        x = []
        y = []
        std = []
        filename = files[index]
        if filename.endswith('.csv'):
            with open(f'{folder}{filename}','r') as csvfile:
                lines = csv.reader(csvfile, delimiter=',')
                title = ""
                for row in lines:
                    if row[1] == 'eval_timesteps':
                        title =  filename.split(".")[0]
                    else:
                        x.append(int(row[1]))
                        y.append(float(row[2]))
                        std.append(float(row[3]))
            items.append({'label':title,'x':x,'y':y,'std':std})
        index +=1
    smoothing_window = 81
    colors = ["#1D9E75", "#D85A30", "#378ADD", "#BA7517"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for item, color in zip(items, colors):
        smooth = savgol_filter(item['y'],smoothing_window,2)
        err = item["std"]/np.sqrt(num_repetitions)
        err_smooth = savgol_filter(err,smoothing_window,2)
        ax.plot(item["x"], smooth, color=color, linewidth=2, label=item["label"])
        ax.fill_between(item["x"],smooth-err_smooth,smooth+err_smooth,alpha=0.2, color=color)
        

    ax.axhline(solved_threshold, color="#E24B4A", linewidth=1.5,
               linestyle="--", label=f"max reward ({solved_threshold})")

    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title("actor-critic training — CartPole-v1")
    ax.legend()
    ax.grid(alpha=0.2)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    plt.show()
 
torch.manual_seed(2001)

#get_full_run_results('REINFORCE', REINFORCE, budget = 1e6)
#get_full_run_results('AC', AC, budget = 1e6)
#get_full_run_results('A2C', A2C, budget = 1e6)
plot_full_runs()



