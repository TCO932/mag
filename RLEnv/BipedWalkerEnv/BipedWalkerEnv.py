import os
import gymnasium as gym
import pybullet as p
import numpy as np
import pybullet_data  # Модуль со встроенными моделями
import time

target = {
    "pos": np.array([0., 3., 0.])
}


class BipedWalkerEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        # Пространство действий (моменты для 6 суставов)
        # -1.57, 1.57
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(6,))

        # Пространство состояний (углы, скорости, положение корпуса и т.д.)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,))
        
        self.render_mode = render_mode
        self.physics_client = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Загрузка модели
        if self.physics_client is None:
            self.physics_client = p.connect(
                p.DIRECT if self.render_mode != "human" else p.GUI)
            
            p.addUserDebugLine(
                target["pos"], 
                [target["pos"][0], target["pos"][1], 3],    # Конечная точка
                lineColorRGB=[1, 0, 0],  # Цвет (R, G, B) от 0 до 1
                lineWidth=2,             # Толщина линии
                lifeTime=0  
            )


            p.setGravity(0, 0, -9.81)

            script_dir = os.path.dirname(os.path.abspath(__file__))
            search_path = os.path.join(script_dir, "URDFs")  # RLEnv/BipedWalkerEnv/URDFs
            p.setAdditionalSearchPath(search_path)

            # Загружаем встроенного гуманоида
            # self.robot = p.loadURDF("biped.urdf")
            self.robot = p.loadURDF("biped_pybullet.urdf")
            self.plane = p.loadURDF("plane.urdf")


        # Сброс робота в начальное положение
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 1.7], [0, 0, 0, 1])
        for joint in range(p.getNumJoints(self.robot)):
            p.resetJointState(self.robot, joint, 0)

        # Возвращаем начальное состояние
        return self._get_obs(), {}

    def step(self, action):
        # Применяем действия (моменты суставов)
        # print(action)
        # action = [1.57, 1.57, 1.57, 1.57, 1.57, 1.57]
        for i in range(p.getNumJoints(self.robot)):  # Пример индексов суставов
            p.setJointMotorControl2(
                self.robot,
                i,
                controlMode=p.TORQUE_CONTROL,
                force=action[i] * 400  # Масштабирование
            )

            # p.setJointMotorControl2(
            #     self.robot,
            #     i,
            #     controlMode=p.POSITION_CONTROL,
            #     targetPosition=action[i] * 1.57,
            #     # force=500,
            #     # positionGain=0.1,
            #     # velocityGain=0.5
            # )

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
        # return np.concatenate([angles, velocities, torso_pos[:2], torso_orn])
        obs = np.concatenate([angles, velocities, torso_pos[:2], torso_orn, target["pos"][:2]])
        return obs

    def _calculate_reward(self, action, joint_velocities):
        # Пример: награда за скорость вперёд
        # torso_vel = p.getBaseVelocity(self.robot)[0][0]  # Скорость по X
        # energy = sum(abs(a * v) for a, v in zip(action, joint_velocities))
        # reward = torso_vel - 0.01 * energy

        torso_pos, _ = p.getBasePositionAndOrientation(self.robot)
        dist = np.linalg.norm(target["pos"][:2])
        current_dist = np.linalg.norm(torso_pos[:2] - target["pos"][:2])

        reward = 1 - current_dist/dist

        return reward

    def _check_done(self):
        # Проверка падения
        torso_pos, _ = p.getBasePositionAndOrientation(self.robot)
        return torso_pos[2] < 0.5

    # def render(self):
    #     print('render')
    #     if self.render_mode == "human":
    #     #     p.setRealTimeSimulation(1)
    #         time.sleep(1/10)  # Замедление для визуализации
    #         print('human')
