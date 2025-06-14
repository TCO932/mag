import time
import gymnasium as gym
import os

import numpy as np
import RegEnvs
import pybullet as p

env_name = "BipedWalkerCustom-v2-stage-4"

script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/BipedWalkerEnv
models_dir = os.path.join(script_dir, "models")  # RLEnv/BipedWalkerEnv/models

# env = gym.make("BipedWalker-v0", render_mode="human")
env = gym.make(env_name, render_mode="human", rtk = 1)
obs, _ = env.reset()

rewards = []
total_rewards = []
step = 0
try:
    while (1):
        step += 1
        action = env.action_space.sample()
        # action = np.zeros(6, dtype=np.float32)
        # print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            if truncated: print("truncated")
            if terminated: print("terminated")
            observation, info = env.reset()
            rewards_sum = np.sum(rewards)
            print(f"Steps: {step}; Rewards sum: {rewards_sum}")
            total_rewards.append(rewards_sum)
            rewards = []
            step = 0
finally:
    # time.sleep(.01)
    print(f"Eps: {len(total_rewards)}; Rewards mean: {np.mean(total_rewards)}")