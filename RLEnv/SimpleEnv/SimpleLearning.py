import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime

# Создаём среду
max_envs = os.cpu_count() 
n_envs = 3
print(f"n_envs: {n_envs}")
env = make_vec_env("LunarLander-v3", n_envs=n_envs)

# Инициализируем PPO
model = PPO(
    "MlpPolicy",         # Нейросеть (MLP)
    env,                 # Среда Gymnasium
    verbose=1,           # Логирование
    tensorboard_log="./logs/LunarLander-v3/",  # Сохранение статистики
    learning_rate=3e-3,  # Скорость обучения
    n_steps=2048,        # Шагов перед обновлением
    batch_size=64,       # Размер батча
)

# Запускаем обучение
model.learn(total_timesteps=50_000, progress_bar=True)

# Сохраняем модель
model.save("models/Lander_ppo_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))