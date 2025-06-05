import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import RegEnvs

env_name = "BipedWalkerCustom-v1"

# Создаём среду с записью видео
env = gym.make(env_name, render_mode="rgb_array")  # или "Ant-v4" для MuJoCo
env = RecordVideo(env, "videos", episode_trigger=lambda x: True)  # записываем каждый эпизод

observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # случайное действие
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()