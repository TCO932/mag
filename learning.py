import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import main
from datetime import datetime

# Создаём среду
max_envs = os.cpu_count() 
n_envs = max_envs
print(f"n_envs: {n_envs}")
env = make_vec_env("BipedWalker-v0", n_envs=n_envs)

# Инициализируем PPO
model = PPO(
    "MlpPolicy",         # Нейросеть (MLP)
    env,                 # Среда Gymnasium
    verbose=1,           # Логирование
    tensorboard_log="./logs/",  # Сохранение статистики
    learning_rate=3e-4,  # Скорость обучения
    n_steps=2048,        # Шагов перед обновлением
    batch_size=64,       # Размер батча
)

# Запускаем обучение
model.learn(total_timesteps=1_000, progress_bar=True)

# Сохраняем модель
model.save("./models/biped_walker_ppo_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
