import time
import gymnasium as gym
from stable_baselines3 import PPO, SAC
import os
from scripts.model_choicer import select
import RegEnvs



env_name = "BipedWalkerCustom-v1"

script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/BipedWalkerEnv
# models_dir = os.path.join(script_dir, "models", "SAC")  # RLEnv/BipedWalkerEnv/models
models_dir = os.path.join(script_dir, "models", "Curriculum", "SAC")  # RLEnv/BipedWalkerEnv/models
os.makedirs(models_dir, exist_ok=True)


max_envs = os.cpu_count() 
n_envs = max_envs
print(f"n_envs: {n_envs}")

model_path = select(models_dir=models_dir, latest=False)

env = gym.make(env_name, render_mode="human")
model = SAC.load(model_path, env=env)       

obs, _ = env.reset()
action, _ = model.predict(obs)
prev_time = time.time()
while (1):
    observation, reward, terminated, truncated, info = env.step(action)
    action, _ = model.predict(observation)
    # print(reward)

    if terminated or truncated:
        observation, info = env.reset()

    if env.render_mode == "human":
        elapsed = time.time() - prev_time
        time_to_wait = (1.0 / 240.0) - elapsed
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        prev_time = time.time()