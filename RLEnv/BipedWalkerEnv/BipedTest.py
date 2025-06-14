import time
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
import os
from scripts.model_choicer import select
import RegEnvs
import argparse


env_name = "BipedWalkerCustom-v2-stage-4"
alg_name = "SAC"
model_name = "BipedWalkerCustom-2_4envs"
model_name = "BipedWalkerCustom-2_1env"
algs = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
seed = 1

def get_args():
    parser = argparse.ArgumentParser(description='Тестирование модели BipedWalker')
    parser.add_argument(
        '--model',
        type=str,
        default=model_name,
        help='Имя модели (без расширения .zip)'
    )
    parser.add_argument(
        '--alg', 
        type=str,
        default=alg_name,
        choices=['PPO', 'SAC', 'TD3'],
        help='Алгоритм (PPO, SAC или TD3)'
    )
    parser.add_argument(
        '-best',
        action='store_true',
        help='Использовать лучшую модель'
    )
    args = parser.parse_args()
    return args.alg, args.model, args.best

alg_name, model_name, use_best = get_args()
print(f"Тестируем модель: {model_name}")
print(f"Алгоритм: {alg_name}")

alg = algs[alg_name]

script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/BipedWalkerEnv
# models_dir = os.path.join(script_dir, "models", "SAC")  # RLEnv/BipedWalkerEnv/models
models_dir = os.path.join(script_dir, "models", "Curriculum", alg_name)  # RLEnv/BipedWalkerEnv/models
os.makedirs(models_dir, exist_ok=True)
# model_path = select(models_dir=models_dir, latest=False)

max_envs = os.cpu_count() 
n_envs = max_envs
print(f"n_envs: {n_envs}")

if use_best:
    model_path=rf"C:\Users\Anton\Desktop\mag\RLEnv\BipedWalkerEnv\models\Curriculum\{alg_name}\best\{model_name}\best_model.zip"
else:
    model_path=rf"C:\Users\Anton\Desktop\mag\RLEnv\BipedWalkerEnv\models\Curriculum\{alg_name}\{model_name}.zip"
    

env = gym.make(env_name, render_mode="human")
model = alg.load(model_path, env=env)       

def do_episode(eps_n=None, seed=None):
    observation, _ = env.reset(seed=None)
    prev_time = time.time()
    eps = 0
    while (eps_n is None or eps < eps_n):
        terminated, truncated = False, False
        while (not (terminated or truncated)):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            # print(reward)

            if terminated or truncated:
                observation, info = env.reset()

            if env.render_mode == "human":
                elapsed = time.time() - prev_time
                time_to_wait = max(0, 1/240 - elapsed)
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                prev_time = time.time()
        eps += 1

do_episode(eps_n=4, seed=seed)