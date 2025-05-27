import gymnasium as gym
from stable_baselines3 import PPO
import os
import glob
from pick import pick

def select(latest, models_dir = "models"):
    os.makedirs(models_dir, exist_ok=True)

    # Получаем список .zip файлов (SB3 сохраняет модели в .zip)
    model_files = glob.glob(os.path.join(models_dir, "*.zip"))

    if not model_files:
        print("В папке 'models' нет сохранённых моделей!")
        exit()

    if latest: 
        selected_model = model_files.pop()
    else:
        # Выбор модели через интерактивное меню
        title = "Выберите модель для загрузки:"
        selected_model = pick(model_files, title, indicator="→")[0]

    print(f"Выбрана {selected_model}")

    return selected_model

# Загружаем модель
model_path = select(latest=True, models_dir = "models")
model = PPO.load(model_path)
env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset()
while (1):
    action, _ = model.predict(obs)
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)

    if terminated or truncated:
        observation, info = env.reset()
