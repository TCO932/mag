import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/SimpleEnv
models_dir = os.path.join(script_dir, "models")  # RLEnv/SimpleEnv/models
os.makedirs(models_dir, exist_ok=True)

# Создаём среду
max_envs = os.cpu_count() 
n_envs = max_envs
print(f"n_envs: {n_envs}")
env = make_vec_env("LunarLander-v3", n_envs=n_envs)

model = PPO.load(models_dir+"/Lander_ppo", env=env, verbose=1)


try:
    model.learn(
        total_timesteps=10_000,  # Новое количество шагов
        progress_bar=True,
        reset_num_timesteps=False,  # Сохраняем предыдущий прогресс
        tb_log_name="Lander_ppo_continued"  # Новый лог для TensorBoard
    )
except KeyboardInterrupt:
    print("Обучение прервано пользователем!")

model.save(models_dir+"/Lander_ppo")