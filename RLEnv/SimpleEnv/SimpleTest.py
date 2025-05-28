import gymnasium as gym
from stable_baselines3 import PPO
import os
from scripts.model_choicer import select


script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/SimpleEnv
models_dir = os.path.join(script_dir, "models")  # RLEnv/SimpleEnv/models

env = gym.make("LunarLander-v3", render_mode="human")
model_path = select(models_dir=models_dir)
model = PPO.load(model_path, env=env)       

obs, _ = env.reset()
action, _ = model.predict(obs)
while (1):
    observation, reward, terminated, truncated, info = env.step(action)
    action, _ = model.predict(observation)
    # print(reward)

    if terminated or truncated:
        observation, info = env.reset()
