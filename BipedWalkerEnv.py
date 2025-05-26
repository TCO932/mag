import gymnasium as gym
import pybullet as p
import numpy as np
import pybullet_data  # Модуль со встроенными моделями
import time


class BipedWalkerEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        # Пространство действий (моменты для 6 суставов)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))
        # Пространство состояний (углы, скорости, положение корпуса и т.д.)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,))
        self.render_mode = render_mode
        self.physics_client = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Загрузка модели
        if self.physics_client is None:
            self.physics_client = p.connect(
                p.DIRECT if self.render_mode != "human" else p.GUI)
            p.setGravity(0, 0, -9.81)

            p.setAdditionalSearchPath(
                pybullet_data.getDataPath())  # Ключевая строка!
            # Загружаем встроенного гуманоида
            # [x,y,z] — позиция спавна
            self.robot = p.loadURDF("biped/biped2d_pybullet.urdf")
            # self.robot = p.loadURDF("biped.urdf")
            self.plane = p.loadURDF("plane.urdf")


        # Сброс робота в начальное положение
        p.resetBasePositionAndOrientation(
            self.robot, [0, 0, 1.0], [0, 0, 0, 1])
        for joint in range(p.getNumJoints(self.robot)):
            p.resetJointState(self.robot, joint, 0)

        # Возвращаем начальное состояние
        return self._get_obs(), {}

    def step(self, action):
        # Применяем действия (моменты суставов)
        for i, joint in enumerate([2, 3, 4, 5, 6, 7]):  # Пример индексов суставов
            p.setJointMotorControl2(
                self.robot,
                joint,
                controlMode=p.TORQUE_CONTROL,
                force=action[i] * 10.0  # Масштабирование
            )

        # Шаг симуляции (240 Гц, но можно проще)
        p.stepSimulation()

        # Получаем новое состояние
        obs = self._get_obs()

        # Считаем награду
        reward = self._calculate_reward(
            action, [state[1] for state in p.getJointStates(self.robot, range(6))])

        # Проверяем терминальное состояние (падение)
        done = self._check_done()

        return obs, reward, done, False, {}

    def _get_obs(self):
        # Пример: углы, скорости, ориентация корпуса
        joint_states = p.getJointStates(self.robot, range(6))
        angles = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot)
        return np.concatenate([angles, velocities, torso_pos[:2], torso_orn])

    def _calculate_reward(self, action, joint_velocities):
        # Пример: награда за скорость вперёд
        torso_vel = p.getBaseVelocity(self.robot)[0][0]  # Скорость по X
        energy = sum(abs(a * v) for a, v in zip(action, joint_velocities))
        reward = torso_vel - 0.01 * energy
        return reward

    def _check_done(self):
        # Проверка падения
        torso_pos, _ = p.getBasePositionAndOrientation(self.robot)
        return torso_pos[2] < 0.5

    def render(self):
        if self.render_mode == "human":
            time.sleep(1/60)  # Замедление для визуализации
