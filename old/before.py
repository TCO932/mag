import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import time

#@title **Функция расчета вознаграждения**
def calculate_reward(robotId, previous_position):
  """
  Вычисляет вознаграждение на основе текущего состояния робота.

  Args:
    robotId: Идентификатор робота в PyBullet.
    previous_position: Предыдущая позиция робота (x, y, z).

  Returns:
    reward: Значение вознаграждения.
    done: True, если эпизод завершен (робот упал), иначе False.
  """

  k_distance = 2.0
  k_speed = 1.5
  k_stability = 1.0
  k_energy = 0 #0.01
  k_fall = 10

  # Вознаграждение за скорость
  current_position, current_orientation = p.getBasePositionAndOrientation(robotId)
  x, y, z = current_position
  vy = (y - previous_position[1]) / (1. / 240.)  # Приблизительная скорость по x
  r_speed = k_speed * vy


  # Вознаграждение за пройденное расстояние
  distance_traveled = y - previous_position[1]  # Разница по Y между текущей и предыдущей позицией
  r_distance = k_distance * distance_traveled

  # Вознаграждение за стабильность
  roll, pitch, yaw = p.getEulerFromQuaternion(current_orientation)
  angle_penalty = k_stability * (roll**2 + pitch**2)
  height_penalty = k_stability * max(0, 0.8 - z)  # Штраф, если высота ниже 0.8
  r_stability = -(angle_penalty + height_penalty)

  # Вознаграждение за энергоэффективность
  num_joints = p.getNumJoints(robotId)
  torques = [p.getJointState(robotId, i)[3] for i in range(num_joints)] # Получаем крутящие моменты
  r_energy = -k_energy * np.sum(np.square(torques))

  # Штраф за падение
  done = False
  if z < 0.5:  # Робот упал, если высота центра масс ниже 0.5
    r_fall = -k_fall
    done = True
  else:
    r_fall = 0

  reward = r_speed + r_distance + r_stability + r_energy + r_fall
  print(f'reward {reward}')
  return reward, done, current_position

#@title **Класс среды WalkingRobotEnv**
class WalkingRobotEnv(gym.Env):
  def __init__(self, render=False, urdf_path="biped.urdf"):
    super(WalkingRobotEnv, self).__init__()
    self.render_mode = render
    try:
      p.disconnect(self.physicsClient) # Попытка закрыть предыдущее подключение
    except:
      pass
    if self.render_mode:
      self.physicsClient = p.connect(p.GUI)
    else:
      self.physicsClient = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    self.planeId = p.loadURDF("plane.urdf")
    # Задаем параметры трения для плоскости
    p.changeDynamics(self.planeId, -1, lateralFriction=0.8, spinningFriction=0.1, rollingFriction=0.001)

    self.robotStartPos = [0, 0, 1.2]
    self.robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    self.urdf_path = urdf_path
    self.robotId = p.loadURDF(self.urdf_path, self.robotStartPos, self.robotStartOrientation)

    # Определение пространства действий (пример для 3 суставов)
    self.action_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

    # Определение пространства наблюдений (пример: углы суставов и высота)
    self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    self.previous_position = self.robotStartPos
    self.num_joints = p.getNumJoints(self.robotId)
    self.max_episode_steps = 1000
    self.current_step = 0

    # Словари для хранения параметров
    self.joint_limits = {
      'right_hip_joint':    (-1.57, 1.57),
      'right_knee_joint':   (-1.57, 1.57),
      'right_ankle_joint':  (-1.57, 1.57),
      'left_hip_joint':     (-1.57, 1.57),
      'left_knee_joint':    (-1.57, 1.57),
      'left_ankle_joint':   (-1.57, 1.57),
    }
    self.joint_ids = {}
    for i in range(self.num_joints):
      joint_info = p.getJointInfo(self.robotId, i)
      joint_name = joint_info[1].decode('UTF-8')
      self.joint_ids[joint_name] = i

    self.interpolation_steps = 10

    # Трение для стоп
    p.changeDynamics(self.robotId, p.getJointInfo(self.robotId, self.joint_ids["right_ankle_joint"])[0], lateralFriction=0.8, spinningFriction=0.1, rollingFriction=0.001)
    p.changeDynamics(self.robotId, p.getJointInfo(self.robotId, self.joint_ids["left_ankle_joint"])[0], lateralFriction=0.8, spinningFriction=0.1, rollingFriction=0.001)

  def step(self, action):
    for joint_name, joint_id in self.joint_ids.items():
      current_angle = p.getJointState(self.robotId, joint_id)[0]
      lower, upper = self.joint_limits[joint_name]

      # Интерполяция действий
      target_angle = lower + action[joint_id] * (upper - lower)
      # print(f'action {action[joint_id]}; target_angle {target_angle}')
      for step in range(self.interpolation_steps):
        intermediate_angle = current_angle + (target_angle - current_angle) * (step + 1) / self.interpolation_steps
        p.setJointMotorControl2(
          self.robotId,
          joint_id,
          p.POSITION_CONTROL,
          targetPosition=intermediate_angle,
          force=160,
          positionGain=0.3,
          velocityGain=0.1
        )


    p.stepSimulation()
    if self.render_mode:
      time.sleep(1. / 240.)

    # Получение наблюдений
    joint_states = [p.getJointState(self.robotId, i)[0] for i in range(self.num_joints)]

    print(f" {self.current_step} Joint Angles:", joint_states)

    base_position, current_orientation = p.getBasePositionAndOrientation(self.robotId)
    roll, pitch, yaw = p.getEulerFromQuaternion(current_orientation)
    normalized_height = (base_position[2] - self.robotStartPos[2]) + 1.0
    observation = np.array(joint_states + [normalized_height, roll, pitch, yaw], dtype=np.float32)

    # Вычисление вознаграждения
    reward, done, self.previous_position = calculate_reward(self.robotId, self.previous_position)

    self.current_step += 1
    if self.current_step >= self.max_episode_steps:
      done = True

    info = {}  # Дополнительная информация (не используется в данном примере)
    return observation, reward, done, False, info

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    self.planeId = p.loadURDF("plane.urdf")
    self.robotId = p.loadURDF(self.urdf_path, self.robotStartPos, self.robotStartOrientation)
    # Проверяем коллизии
    closest_points = p.getClosestPoints(self.robotId, self.planeId, distance=0.01) # Ищем коллизии с плоскостью
    if closest_points:
      print("Initial collision detected!")
    self.previous_position = self.robotStartPos
    self.current_step = 0

    # Получение начальных наблюдений
    joint_states = [p.getJointState(self.robotId, i)[0] for i in range(self.num_joints)]
    base_position, current_orientation = p.getBasePositionAndOrientation(self.robotId)
    roll, pitch, yaw = p.getEulerFromQuaternion(current_orientation)
    normalized_height = (base_position[2] - self.robotStartPos[2]) + 1.0
    observation = np.array(joint_states + [normalized_height, roll, pitch, yaw], dtype=np.float32)
    info = {}
    return observation, info

  def close(self):
    p.disconnect()