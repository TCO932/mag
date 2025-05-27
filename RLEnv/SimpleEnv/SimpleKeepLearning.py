import glob
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime

def select(latest, models_dir = "./models"):
    os.makedirs(models_dir, exist_ok=True)

    # Получаем список .zip файлов (SB3 сохраняет модели в .zip)
    model_files = glob.glob(os.path.join(models_dir, "*.zip"))

    if not model_files:
        print("В папке 'models' нет сохранённых моделей!")
        exit()

    selected_model = model_files.pop()
    print(f"Выбрана {selected_model}")

    return selected_model

# Создаём среду
max_envs = os.cpu_count() 
n_envs = max_envs
print(f"n_envs: {n_envs}")
env = make_vec_env("LunarLander-v3", n_envs=n_envs)

# Инициализируем PPO
model = PPO(
    "MlpPolicy",         # Нейросеть (MLP)
    env,                 # Среда Gymnasium
    verbose=1,           # Логирование
    tensorboard_log="./logs/LunarLander-v3/",  # Сохранение статистики
    learning_rate=3e-4,  # Скорость обучения
    n_steps=2048,        # Шагов перед обновлением
    batch_size=64,       # Размер батча
)

# Запускаем обучение
model.learn(total_timesteps=100_000, progress_bar=True)

# Сохраняем модель
model.save("../models/Lander_ppo_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

model.learn(
    total_timesteps=500_000,  # Новое количество шагов
    progress_bar=True,
    reset_num_timesteps=False,  # Сохраняем предыдущий прогресс
    tb_log_name="PPO_BipedWalker_continued"  # Новый лог для TensorBoard
)