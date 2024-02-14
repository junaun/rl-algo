from stable_baselines3 import PPO
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from super_ppo import SUPERPPO
from super_a2c import SuperActorCriticPolicy
import pandas as pd
import numpy as np
from plot import plot_result
from tqdm import tqdm

if __name__ == '__main__':
    # Create environment
    # env_id = "BipedalWalker-v3"
    env_id = "CartPole-v1"
    total_timesteps = int(30e3)
    seed = 2024

    env_1 = make_vec_env(env_id=env_id, n_envs=1, vec_env_cls=SubprocVecEnv, seed=seed )
    env_1 = VecMonitor(env_1, filename="teacher")
    env_2 = make_vec_env(env_id=env_id, n_envs=1, vec_env_cls=SubprocVecEnv, seed=seed )
    env_2 = VecMonitor(env_2, filename="student")
    env_3 = make_vec_env(env_id=env_id, n_envs=1, vec_env_cls=SubprocVecEnv, seed=seed )
    env_3 = VecMonitor(env_3, filename="dummy")

    teacher = PPO("MlpPolicy", env_1, verbose=0, seed=seed)
    student = SUPERPPO(SuperActorCriticPolicy, env_2, verbose=0, teacher_model=teacher,
                                imit_coeff=1.0, seed=seed)
    dummy = PPO(SuperActorCriticPolicy, env_3, verbose=0, seed=seed)

    for _ in tqdm(range(10)):
        batch = int(total_timesteps / 10)
        dummy.learn(total_timesteps=batch, progress_bar=False)
        teacher.learn(total_timesteps=batch, progress_bar=False)
        student.teacher_model = teacher
        student.learn(total_timesteps=batch, progress_bar=False)

    plot_result()