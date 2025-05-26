import gymnasium as gym
import model_choicer
from stable_baselines3 import PPO
import main

# Загружаем модель
model_path = model_choicer.select(latest=True)
model = PPO.load(model_path)
env = gym.make("BipedWalker-v0", render_mode="human")
obs, _ = env.reset()
while (1):
    action, _ = model.predict(obs)
    obs, _, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()

