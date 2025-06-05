import time
import gymnasium as gym
import os
import RegEnvs
import pybullet as p

env_name = "BipedWalkerCustom-v1"

script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/BipedWalkerEnv
models_dir = os.path.join(script_dir, "models")  # RLEnv/BipedWalkerEnv/models

# env = gym.make("BipedWalker-v0", render_mode="human")
env = gym.make(env_name, render_mode="human")
obs, _ = env.reset()

cam_upd = env.cam_init()
while (1):
    cam_upd()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        pass

    # time.sleep(.01)