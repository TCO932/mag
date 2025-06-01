import time
import gymnasium as gym
import os

import numpy as np
import RegEnvs


script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/BipedWalkerEnv
models_dir = os.path.join(script_dir, "models")  # RLEnv/BipedWalkerEnv/models

# env = gym.make("BipedWalker-v0", render_mode="human")
env = gym.make("BipedWalkerCustom-v1", render_mode="human")
obs, _ = env.reset()

n = 1_000
sim_drts = np.array([])
prev_time = time.time()
for i in range(n):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # print(info["sim_drt"])
    sim_drts = np.append(sim_drts, info["sim_drt"])

    
    if env.render_mode == "human":
        elapsed = time.time() - prev_time
        time_to_wait = (1.0 / 240.0) - elapsed  # Цель: 60 FPS
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        prev_time = time.time()

print("mean: ", np.mean(sim_drts))
print("mean: ", np.mean(sim_drts)*240)
