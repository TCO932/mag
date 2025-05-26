import gymnasium as gym
import pybullet as p
import time
import main

env = gym.make("BipedWalker-v0", render_mode="human")
observation, info = env.reset()

# # Получаем ID клиента PyBullet (если среда использует PyBullet)
# if hasattr(env, 'physics_client'):
#     physics_client = env.physics_client
# else:
#     # Для некоторых сред PyBullet (например, из pybullet_envs)
#     physics_client = env.unwrapped.client_id  # или env.unwrapped.client_id

# Добавляем слайдеры в GUI (если рендеринг через PyBullet)
gravity_slider = p.addUserDebugParameter("Gravity", -50, 50, -10)

while True:
    # Читаем значения слайдеров
    gravity = p.readUserDebugParameter(gravity_slider)

    # Применяем параметры к миру (если возможно)
    p.setGravity(0, 0, gravity)  # Изменяем гравитацию
    # p.changeDynamics(env.unwrapped.plane, -1, lateralFriction=friction)

    # Шаг среды
    action = env.action_space.sample()  # или модель RL
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated:
        observation, info = env.reset()

    # time.sleep()  # Замедление для визуализации