import gymnasium as gym

from PPG import PPG
from AuxPolicy import AuxActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

# Create environment
env = gym.make("CartPole-v1")
env = Monitor(env, filename="ppg")

# Instantiate the agent
model = PPG(AuxActorCriticPolicy, env, verbose=1, learning_rate=5e-4)
# model = PPO("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5), progress_bar=True)
exit()