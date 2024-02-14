import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_result():
    # filenames = ['teacher.monitor.csv'] + glob.glob('student.monitor.csv')
    filenames = ['teacher.monitor.csv', 'student.monitor.csv', 'dummy.monitor.csv']
    fig, ax = plt.subplots(figsize=(6, 4))
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
        label = filename.split('.m')[0]

        # Plotting
        ax.plot(smoothed_epochs, smoothed_return, label=label)

    # Setting labels, title, legend, etc.
    ax.set_xlabel('training epochs')
    ax.set_ylabel('episode return')
    ax.set_title('Training')  # Adjust the title for each graph.
    ax.legend()

    # plt.show()
    plt.savefig('result.png')