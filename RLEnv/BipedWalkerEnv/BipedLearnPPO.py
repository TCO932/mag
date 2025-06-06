import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import RegEnvs
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


env_name = "BipedWalkerCustom-v2"
model_name = "BipedWalkerCustom-v4"
alg = "PPO"
total_timesteps = 5_000_000

script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/BipedWalkerEnv
models_dir = os.path.join(script_dir, "models", alg)  # RLEnv/BipedWalkerEnv/models/PPO
os.makedirs(models_dir, exist_ok=True)
tb_logs_path = os.path.join(script_dir, "logs", model_name, alg)

cores = os.cpu_count() 
training_period = 20_000
n_envs = 4
print(f"n_envs: {n_envs}")

# env = make_vec_env(env_name, n_envs=n_envs)
# env = VecNormalize(env, norm_obs=True, norm_reward=True) # Нормализуем среду
env = gym.make(env_name, render_mode="human", rtk=0)
# env = RecordVideo(env, video_folder=script_dir+"/videos/"+model_name, name_prefix="training",
#                   episode_trigger=lambda x: x % training_period == 0)

# Загрузка или создание модели
try:
    model = PPO.load(models_dir+"/"+model_name, env=env, verbose=1)
    print("Модель загружена!")
except (ValueError, FileNotFoundError):
    print("Создаём новую модель...")
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     n_steps=512,          # Уменьшил для более частых обновлений
    #     batch_size=64,
    #     learning_rate=3e-3,    # Более стабильное значение
    #     ent_coef=0.01,        # Поощрение исследования
    #     gamma=0.99,           # Фактор дисконтирования
    #     gae_lambda=0.95,      # Параметр GAE
    #     max_grad_norm=0.5,    # Ограничение градиентов
    #     clip_range=0.2,       # Стандартный для PPO
    #     n_epochs=10,          # Количество эпох оптимизации
    #     tensorboard_log=os.path.join(script_dir, "logs", model_name),
    # )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,           # Увеличить для более стабильных updates
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
    model.learn(
        total_timesteps=total_timesteps,  # Новое количество шагов
        progress_bar=True,
        reset_num_timesteps=False,  # Сохраняем предыдущий прогресс
        tb_log_name=tb_logs_path  # Новый лог для TensorBoard
    )
except KeyboardInterrupt:
    print("Обучение прервано пользователем!")

model.save(models_dir+"/"+model_name)