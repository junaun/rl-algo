from stable_baselines3 import PPO
from super_ppo import SUPERPPO
from super_a2c import SuperActorCriticPolicy
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
import pandas as pd
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import glob
import gymnasium as gym
import torch as th

if __name__ == '__main__':
    # Create environment
    env_id = "InvertedPendulum-v4"
    total_timesteps = int(1e5)
    seed = 2024

    env = make_vec_env(env_id=env_id, n_envs=16, vec_env_cls=SubprocVecEnv, seed=seed*3)
    env = VecMonitor(env, filename="teacher")

    # Instantiate the agent
    teacher_model = PPO("MlpPolicy", env, verbose=1, seed=seed*3)
    # Train the agent and display a progress bar
    teacher_model.learn(total_timesteps=total_timesteps, progress_bar=True)
    teacher_model.save("teacher-model-pendulum")
    del teacher_model

    env = make_vec_env(env_id=env_id, n_envs=16, vec_env_cls=SubprocVecEnv, seed=seed)
    env = VecMonitor(env, filename="student_without_imitation")

    # Instantiate the agent
    model = PPO(SuperActorCriticPolicy, env, verbose=1, seed=seed)
    # Train the agent and display a progress bar
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("student-model-pendulum")
    del model

    for coeff in [1.0]:
        # Instantiate the agent
        # Create environment
        env = make_vec_env(env_id=env_id, n_envs=16, vec_env_cls=SubprocVecEnv, seed=seed)
        teacher_model = PPO.load("teacher-model-pendulum.zip", env = env)
        env = VecMonitor(env, filename=f"student_imitation_{coeff}")
        student_model = SUPERPPO(SuperActorCriticPolicy, env, verbose=1, teacher_model=teacher_model,
                                    imit_coeff=coeff, seed=seed)
        # Train the agent and display a progress bar
        student_model.learn(total_timesteps=total_timesteps, progress_bar=True)
        # model.save("student-model")

        # mean_reward, std_reward = evaluate_policy(teacher_model, env, n_eval_episodes=10)
        # print(f"Teacher's Mean reward = {mean_reward} +/- {std_reward}")
        # mean_reward, std_reward = evaluate_policy(student_model, env, n_eval_episodes=10)
        # print(f"Student's Mean reward = {mean_reward} +/- {std_reward}")

    filenames = ['teacher.monitor.csv'] + glob.glob('student_*.monitor.csv')
    print(filenames)

    # Load data from csv files and plot
    # sns.set_style("whitegrid")
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
        window_size = 300  # Adjust this based on your preference
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
    ax.set_title('mujoco_pendulum')  # Adjust the title for each graph.
    ax.legend()

    # plt.show()
    plt.savefig('result.png')
