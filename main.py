import pybullet as p
from BipedWalkerEnv import BipedWalkerEnv  # Кастомная среда

env = BipedWalkerEnv("human")

import gymnasium as gym
from gymnasium.envs.registration import register  

register(  
    id="BipedWalker-v0",  
    entry_point="BipedWalkerEnv:BipedWalkerEnv",  # Путь к классу  
    max_episode_steps=1000,  
)  

# Теперь можно создать среду:  
env = gym.make("BipedWalker-v0", render_mode="human")