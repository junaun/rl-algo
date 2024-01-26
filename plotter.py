import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import numpy as np

# List of csv filenames
# filenames = ['A2C.monitor.csv', 'DDPG.monitor.csv','PPO.monitor.csv','SAC.monitor.csv']
filenames = ['ppg.monitor.csv', 'ppo.monitor.csv']

# Load data from csv files and plot
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(6, 4))

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

for filename in filenames:
    # Load data
    data = pd.read_csv(filename, skiprows=1, header=None)[1:]
    # training_epochs = data['training_epochs'].values
    episode_return = data[0].values.astype(float)
    training_epochs = list(range(len(episode_return)))

    # Using a moving average to smooth the curve
    window_size = 30  # Adjust this based on your preference
    smoothed_return = moving_average(episode_return, window_size)

    # Adjust epochs to match the length of smoothed_return
    smoothed_epochs = training_epochs[:len(smoothed_return)]

    # Extract label from filename (remove '.csv' part)
    label = filename.split('.')[0]

    # Plotting
    ax.plot(smoothed_epochs, smoothed_return, label=label)

# Setting labels, title, legend, etc.
ax.set_xlabel('training epochs')
ax.set_ylabel('episode return')
ax.set_title('PPG vs PPO env:CartPole-v1')  # Adjust the title for each graph.
ax.legend()

plt.show()
