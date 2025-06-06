import os
import gymnasium as gym

import inquirer
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import RegEnvs
import BipedCurriculumCallback as cb 


env_name = "BipedWalkerCustom-v1"
model_name = "BipedWalkerCustom-v4"
alg_name = "SAC"
algs = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
alg = algs[alg_name]

total_timesteps = 2_000_000
n_envs = 4

if __name__ == "__main__":
    algs_names = list(algs.keys())
    choice = inquirer.prompt([
        inquirer.List("algorithm", message="Select algorithm", choices=algs_names, default=algs_names[1])
    ])
    alg_name = choice["algorithm"]
    alg = algs[alg_name]
    print(f"Using {alg_name}...")

script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/BipedWalkerEnv
models_dir = os.path.join(script_dir, "models", "Curriculum", alg_name)  # RLEnv/BipedWalkerEnv/models/PPO
os.makedirs(models_dir, exist_ok=True)
tb_logs_path = os.path.join(script_dir, "logs", "Curriculum", model_name, alg_name)

# Separate evaluation env
# eval_env = gym.make(env_name)
# eval_env.env_method("set_phase", 0)

# Stop training when the model reaches the reward threshold
# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=90, verbose=1)
# eval_callback = EvalCallback(
#     eval_env, 
#     callback_on_new_best=callback_on_best,
#     best_model_save_path=models_dir+"/best/",
#     verbose=1
# )

# model = SAC(
#     "MlpPolicy",
#     eval_env,
#     verbose=1,
#     # n_steps=2048,           # Увеличить для более стабильных updates
#     # batch_size=64,          # Можно попробовать увеличить до 128-256
#     # learning_rate=3e-4,   # Стандартное значение для PPO
#     # ent_coef=0.05,           # Увеличить для большего исследования
#     # gamma=0.99,
#     # gae_lambda=0.95,
#     # max_grad_norm=0.5,
#     # clip_range=0.1,         # Уменьшить для более консервативных обновлений
#     # n_epochs=10,
#     tensorboard_log=tb_logs_path,
# )
# for i in range(3):
#     eval_env.env_method("set_phase", i)
#     # Almost infinite number of timesteps, but the training will stop
#     # early as soon as the reward threshold is reached
#     model.learn(
#         total_timesteps=int(1e10),
#         progress_bar=True,
#         reset_num_timesteps=False,  # Сохраняем предыдущий прогресс
#         tb_log_name=tb_logs_path, 
#         callback=eval_callback
#     )
#     model.save(models_dir+"/"+model_name+f"phase_{i}")

# env = make_vec_env("BipedalEnv-v0", n_envs=4)
# model = PPO("MlpPolicy", env, verbose=1)

env = make_vec_env(
    env_name,
    n_envs=n_envs,
)
env.render_mode = "human"
env = VecNormalize(env, norm_obs=True, norm_reward=True) # Нормализуем среду
# env = gym.make(env_name, )
try:
    model = alg.load(models_dir+"/"+model_name, env=env, verbose=1)
    print("Модель загружена!")
except (ValueError, FileNotFoundError):
    print("Создаём новую модель...")
    model = alg(
        "MlpPolicy",
        env,
        verbose=1,
        # n_steps=2048,           # Увеличить для более стабильных updates
        # batch_size=64,          # Можно попробовать увеличить до 128-256
        # learning_rate=3e-4,   # Стандартное значение для PPO
        # ent_coef=0.05,           # Увеличить для большего исследования
        # gamma=0.99,
        # gae_lambda=0.95,
        # max_grad_norm=0.5,
        # clip_range=0.1,         # Уменьшить для более консервативных обновлений
        # n_epochs=10,
        tensorboard_log=tb_logs_path,
    )

try:
    callback = cb.CurriculumCallback(env, reward_threshold=500)
    model.learn(
        total_timesteps=total_timesteps, 
        progress_bar=True,
        callback=callback
    )
except KeyboardInterrupt:
    print("Обучение прервано пользователем!")

model.save(models_dir+"/"+model_name)