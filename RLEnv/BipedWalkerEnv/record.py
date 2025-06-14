# main_script_opencv.py
import os
import cv2
import numpy as np
import RegEnvs
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3

def main():
    env_name = "BipedWalkerCustom-v2-stage-4"
    alg_name = "SAC"
    algs = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
    alg = algs[alg_name]

    script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/BipedWalkerEnv
    # models_dir = os.path.join(script_dir, "models", "SAC")  # RLEnv/BipedWalkerEnv/models
    models_dir = os.path.join(script_dir, "models", "Curriculum", alg_name)  # RLEnv/BipedWalkerEnv/models
    os.makedirs(models_dir, exist_ok=True)
    # model_path = select(models_dir=models_dir, latest=False)

    max_envs = os.cpu_count() 
    n_envs = max_envs
    print(f"n_envs: {n_envs}")

    model_path=rf"C:\Users\Anton\Desktop\mag\RLEnv\BipedWalkerEnv\models\Curriculum\{alg_name}\best\best_model.zip"


    env = gym.make(env_name, render_mode="rgb_array")
    observation, info = env.reset()
    model = alg.load(model_path, env=env)       

    # Настройки видео
    video_path = "videos/biped_walker_opencv.mp4"
    height, width, _ = env.render().shape
    # 'mp4v' - это кодек. Может отличаться для разных систем (e.g., 'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(video_path, fourcc, env.metadata['render_fps'], (width, height))

    try:
        for i in range(1000):
            print(i)
            action, _ = model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Получаем кадр
            frame = env.render()
            
            # OpenCV использует формат BGR, а PyBullet/Gym возвращают RGB. Конвертируем.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Записываем кадр в видеофайл
            video_writer.write(frame_bgr)

            if terminated or truncated:
                observation, info = env.reset()
    finally:
        # Важно! Освобождаем ресурсы
        env.close()
        video_writer.release()
        print(f"Видео сохранено в: {video_path}")

if __name__ == '__main__':
    main()