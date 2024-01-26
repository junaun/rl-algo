import gymnasium as gym

from PPG import PPG
from AuxPolicy import AuxActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import torch as th

# Create environment
if __name__ == '__main__':
    # cart = gym.make("CartPole-v1")
    env = make_vec_env(env_id="CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv, seed=2024 )
    env = VecMonitor(env, filename="ppg")
    # Instantiate the agent
    model = PPG(AuxActorCriticPolicy, env, verbose=1, learning_rate=5e-4, device="cpu", policy_kwargs=dict(activation_fn=th.nn.Identity))
    # model = PPO("MlpPolicy", env, verbose=1, learning_rate=5e-4, n_epochs=32, batch_size=64, device="cpu")
    # model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e5), progress_bar=True)
    exit()