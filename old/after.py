import os
import gymnasium as gym
import pybullet as p
import numpy as np
import pybullet_data  # Модуль со встроенными моделями
import time

target = {
    "pos": np.array([-1., 3., 0.]),
    "id": None
}
max_force = 100

class BipedWalkerEnv(gym.Env):

    def __init__(self, render_mode=None, rtk = 0):
        super().__init__()
        # Пространство действий (моменты для 6 суставов)
        # -1.57, 1.57
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(6,), dtype=np.float32)
        # Пространство состояний (углы, скорости, положение корпуса и т.д.)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.rtk = rtk
        self.step_count = 0

        # Загрузка модели
        self.physics_client = p.connect(
            p.DIRECT if self.render_mode != "human" else p.GUI)
            
        p.setGravity(0, 0, -9.81)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_path = os.path.join(script_dir, "URDFs")  # RLEnv/BipedWalkerEnv/URDFs
        p.setAdditionalSearchPath(search_path)

        # Загружаем встроенного гуманоида
        self.robot = p.loadURDF("biped.urdf")
        # self.robot = p.loadURDF("biped_pybullet.urdf")
        self.plane = p.loadURDF("plane.urdf")

        self.metadata = {
            "render_modes": ["human", "rgb_array"],  # Explicitly list supported modes
            "render_fps": 30,  # Optional: framerate for rendering
        }
        
        if self.render_mode == "human":
            self.create_toggle_btn()
            self.create_joint_sliders()

        self.debug()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Сброс робота в начальное положение
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 1.6], [0, 0, 0, 1])
        for joint in range(p.getNumJoints(self.robot)):
            p.resetJointState(self.robot, joint, 0)

        self.step_count = 0
        XY_pos = np.random.randint(2, 5, size=2)
        X_sign = np.random.choice([-1, 1], size=1)
        XY_signs = np.append(X_sign, [1])
        target["pos"] = np.append(XY_pos * XY_signs, [0])


        if not target["id"]:
            target["id"] = p.addUserDebugLine(
                target["pos"], 
                [target["pos"][0], target["pos"][1], 3],    # Конечная точка
                lineColorRGB=[1, 0, 0],  # Цвет (R, G, B) от 0 до 1
                lineWidth=2,             # Толщина линии
            )
        else:
            p.addUserDebugLine(
                target["pos"], 
                [target["pos"][0], target["pos"][1], 3],    # Конечная точка
                lineColorRGB=[1, 0, 0],  # Цвет (R, G, B) от 0 до 1
                lineWidth=2,             # Толщина линии
                replaceItemUniqueId=target["id"]  
            )
            

        # Возвращаем начальное состояние
        return self._get_obs(), {}

    def step(self, action):
        time_before = time.time()
        # Применяем действия (моменты суставов)
        # print(action)
        # action = [1.57, 1.57, 1.57, 1.57, 1.57, 1.57]
        for i in range(p.getNumJoints(self.robot)):
            p.setJointMotorControl2(
                self.robot,
                i,
                controlMode=p.TORQUE_CONTROL,
                force=action[i] * 250
            )

        p.stepSimulation()

        # Получаем новое состояние
        obs = self._get_obs()

        # Считаем награду

        # Проверяем терминальное состояние (падение)
        sim_drt = time.time() - time_before

        reward, target_reached = self._calculate_reward(action)
        
        falled = self._check_fall()
        reward -= 50 if falled else 0
        reward += 80 if target_reached else 0

        done = np.logical_or(
            falled,
            target_reached
        )
        
        if self.render_mode == "human":
            if hasattr(self, 'joint_sliders'):
                self.update_joints_from_sliders()

            if self.rtk > 0:
                time_to_wait = (self.rtk / 240.0) - sim_drt
                if time_to_wait > 0:
                    time.sleep(time_to_wait)

        self.step_count += 1

        return obs, reward, done, False, {"sim_drt": sim_drt}

    def _get_obs(self):
        # Пример: углы, скорости, ориентация корпуса
        joint_states = p.getJointStates(self.robot, range(6))

        angles = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot)
        deviation_angle = self.deviation_angle(torso_pos, torso_orn)
        # return np.concatenate([angles, velocities, torso_pos[:2], torso_orn])
        obs = np.concatenate([
            angles, 
            velocities, 
            torso_pos[:2], 
            torso_orn, 
            [deviation_angle],

        ], dtype=np.float32)
        return obs
    
    def _calculate_reward(self, action):
        done = False
        # Получаем текущую позицию и ориентацию, и скорость
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot)
        torso_vel, _ = p.getBaseVelocity(self.robot)
        
        # 1. Награда за приближение к цели (основная компонента)
        target_pos = np.array(target["pos"][:2], dtype=np.float32)  # Координаты X,Y цели
        current_pos = np.array(torso_pos[:2], dtype=np.float32)
        displacement = target_pos - current_pos
        current_dist = np.linalg.norm(displacement)
        
        # Если dist - начальное расстояние, вычисляем его один раз в начале эпизода
        if not hasattr(self, 'initial_dist'):
            self.initial_dist = current_dist
        
        # Нормированная награда за сокращение расстояния
        distance = (self.initial_dist - current_dist) / self.initial_dist
        velocity = np.linalg.norm(torso_vel[:3])
        # if self.step_count % 240 == 0: print(velocity)
        done = current_dist < 0.2 and velocity < 0.9
        distance_reward = distance ** 3
        
        # 2. Штраф за энергию (резкие движения)
        energy_penalty = -0.0001 * np.sum(np.square(action))
        
        # 3. Штраф за отклонение от вертикали (если нужно избегать падения)
        # (quaternion в torso_orn: [x,y,z,w])
        upright_penalty = -0.00001 * (1 - torso_orn[3])  # Используем w-компоненту кватерниона
        
        # 4. Поощрение за скорость к цели (опционально)
        velocity_reward = 0.01 * np.dot(torso_vel[:2], displacement / (current_dist + 1e-5))
        flight_penalty = self.calc_flight_penalty()
        stability_reward = self.exp_reward(self.step_count, scale=0.001)
        deviation_angle_penalty = -np.abs(action[-1]*2, dtype=np.float32)
        # Итоговая награда
        reward = np.sum([
            distance_reward, 
            velocity_reward,
            energy_penalty, 
            upright_penalty,
            flight_penalty,
            stability_reward,
            deviation_angle_penalty
        ], dtype=np.float32)
        return reward, done

    def _check_fall(self):
        # Проверка падения
        torso_pos, _ = p.getBasePositionAndOrientation(self.robot)
        return torso_pos[2] < 0.5

    # def render(self):
    #     print('render')
    #     if self.render_mode == "human":
    #     #     p.setRealTimeSimulation(1)
    #         time.sleep(1/10)  # Замедление для визуализации
    #         print('human')

    def create_toggle_btn(self):
        self.toggle_btn = p.addUserDebugParameter("Toggle controls", 1, 0, 0)

    def create_joint_sliders(self):
        self.joint_sliders = {}
        num_joints = p.getNumJoints(self.robot)
        
        for joint_idx in range(num_joints):
            joint_info = p.getJointInfo(self.robot, joint_idx)
            joint_name = joint_info[1].decode('utf-8')
            
            # Получаем ограничения сустава
            lower_limit = joint_info[8]  # Минимальное положение
            upper_limit = joint_info[9]  # Максимальное положение
            
            # Создаем слайдер только для вращательных суставов (JOINT_REVOLUTE)
            if joint_info[2] == p.JOINT_REVOLUTE:
                slider = p.addUserDebugParameter(
                    paramName=joint_name,
                    rangeMin=lower_limit,
                    rangeMax=upper_limit,
                    startValue=0  # Начальное положение
                )
                self.joint_sliders[joint_idx] = slider

    def update_joints_from_sliders(self):
        toggle_btn_value = p.readUserDebugParameter(self.toggle_btn)
        if (toggle_btn_value % 2):
            for joint_idx, slider_id in self.joint_sliders.items():
                slider_value = p.readUserDebugParameter(slider_id)
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=slider_value,
                    force=200
                )

    def calc_flight_penalty(self):
        # ... предыдущий код ...
        
        # 4. Штраф за прыжки/полёт (проверка контакта с землёй)
        flight_penalty = 0.0
        foots_on_ground = 0
        
        # Для каждой ступни (пример для 4-ногого робота)
        for foot_link_id in [2, 5]:  # ID link'ов ступней (уточните для вашей модели)
            contact_info = p.getContactPoints(
                bodyA=self.robot,
                linkIndexA=foot_link_id,
                bodyB=self.plane  # ID плоскости (обычно 0)
            )
            if (len(contact_info)):
                foots_on_ground += 1 
            
        is_in_air = foots_on_ground == 0
        
        # Штрафуем только если робот поднялся слишком высоко
        torso_pos, _ = p.getBasePositionAndOrientation(self.robot)
        if is_in_air and torso_pos[2] > 2.0:  # Порог высоты (настройте под ваш робот)
            flight_penalty = -2.0  # Жёсткий штраф за полёт
        elif is_in_air:
            flight_penalty = -0.6  # Мягкий штраф за отрыв
        
        return flight_penalty
    
    def exp_reward(self, value: np.float32, max_reward=10.0, scale=0.001):
        return np.min([max_reward, np.exp(scale * value) - 1])
    
    def deviation_angle(self, torso_pos, torso_orn, dbg=False):
        # Вычисляем угол отклонения
        rotation_matrix = np.array(p.getMatrixFromQuaternion(torso_orn), dtype=np.float32).reshape(3, 3)
        forward_vector = rotation_matrix[:, 1]
        target_vector = np.array(target["pos"][:2]) - np.array(torso_pos[:2],dtype=np.float32)
        deviation_angle = self.angle_between_vectors(forward_vector[:2], target_vector[:2])

        if dbg:
            # Вектор "вперёд" (красный)
            p.addUserDebugLine(
                torso_pos, 
                [torso_pos[0] + forward_vector[0], 
                  torso_pos[1] + forward_vector[1], 
                  torso_pos[2]], 
                [1, 0, 0], 
                2, 
                lifeTime=1./240.
            )

            # Вектор к цели (зелёный)
            p.addUserDebugLine(
                torso_pos, 
                [target["pos"][0], target["pos"][1], torso_pos[2]], 
                [0, 1, 0], 
                2, 
                lifeTime=1./240.
            )

        return deviation_angle
    
    def angle_between_vectors(self, v1: np.float32, v2: np.float32):
        # Нормализуем вектора
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Скалярное произведение и угол
        dot_product = np.dot(v1_norm, v2_norm)
        cross_product = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]  # z-компонента векторного произведения
        
        angle = np.arctan2(cross_product, dot_product)
        
        return angle  # Возвращаем угол в радианах
    
    def debug(self):
        times = 1

        if times < 1:
            for i in range(p.getNumJoints(self.robot)):
                info = p.getJointInfo(self.robot, i)
                print(f"Joint ID: {i}, Name: {info[1].decode('utf-8')}, Link: {info[12].decode('utf-8')}")
                

            times =+ 1
