"""
Experiment file used to get the averaged results of multiple runs of REINFORCE, AC and A2C, creating csv files with the averages
and plotting themm together.
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
    """
    Executes the optimization process for the correspondent type of ModelFreeLearner, and generates a csv file with the averaged resuls.
    """  
    # specify the directory name
    directory_name = 'Full_Run_Results'

    # create the directory
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
    
    #loop over number of repetitions for averaging
    for i in range(num_repetitions):
        curr_env = gym.make("CartPole-v1")
        #Initialize learner depending on the experiment
        learner = learner_type(curr_env,2,2,0.99, actor_lr,critic_lr)
        #Get optimization results
        evaluation = learner.optimize(budget)
        if len(evaluation) > (budget/250):
            evaluation = evaluation[:-int(len(evaluation) - (budget/250))]
        curr_eval_returns, curr_eval_timesteps  = [*zip(*evaluation)]
        
        results.append(np.array(curr_eval_returns))
    
    results = np.array(results)
    print(results)
    print(curr_eval_timesteps)
    #export evaluation to a csv file
    df = pd.DataFrame({
        "eval_timesteps": list(map(int, curr_eval_timesteps)),
        "eval_mean_returns":np.mean(results, axis=0),
        "eval_std_returns":np.std(results, axis=0)
        })
    df.to_csv(file_path)

def plot_full_runs(solved_threshold=500, num_repetitions = 5):
    """
    Goes through the Full_Run_Results folder and plots the results of each csv file in it
    """
    items = []
    folder = f'Full_Run_Results/'
    files = os.listdir(folder)
    index = 0

    #iterate through the files getting the file name for the plot title, timesteps, results and standard deviation, saving it as an item
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
    #define smoothing window
    smoothing_window = 81
    colors = ["#1D9E75", "#D85A30", "#378ADD", "#BA7517"]
    fig, ax = plt.subplots(figsize=(10, 5))
    #plot for every item collected, adding smoothing with a savgol_filter
    for item, color in zip(items, colors):
        smooth = savgol_filter(item['y'],smoothing_window,2)
        err = item["std"]/np.sqrt(num_repetitions)
        err_smooth = savgol_filter(err,smoothing_window,2)
        ax.plot(item["x"], smooth, color=color, linewidth=2, label=item["label"])
        ax.fill_between(item["x"],smooth-err_smooth,smooth+err_smooth,alpha=0.2, color=color)
        
    #plot max reward threshold
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

#select a seed to make results replicable
torch.manual_seed(2001)

#get_full_run_results('REINFORCE', REINFORCE, budget = 1e6)
#get_full_run_results('AC', AC, budget = 1e6)
#get_full_run_results('A2C', A2C, budget = 1e6, critic_lr = 1e-4)
plot_full_runs()



