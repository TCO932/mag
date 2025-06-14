import os
import gymnasium as gym
import pybullet as p
import numpy as np
import pybullet_data  # Модуль со встроенными моделями
import time


class Target():
    def __init__(self, pos=np.array([0., 3., 0.]), seed=None):
        self.pos = pos
        self.id = p.addUserDebugLine(
            self.pos, 
            [self.pos[0], self.pos[1], 3],      # Конечная точка
            lineColorRGB=[1, 0, 0],             # Цвет (R, G, B) от 0 до 1
            lineWidth=2,                        # Толщина линии
        )
        self.rng = np.random.RandomState(seed)

    def randomize_target_pos(self, max_dist=5, angle1=-np.pi,angle2=np.pi):
        angle1 = angle1 + np.pi/2 # угол от оси Y
        angle2 = angle2 + np.pi/2 # угол от оси Y
        # dist = np.random.randint(2, max_dist)
        dist = 4 
        angle = self.rng.uniform(angle1, angle2)
        self.pos = np.array([dist * np.cos(angle), dist * np.sin(angle), 0])

        p.addUserDebugLine(
            self.pos, 
            [self.pos[0], self.pos[1], 3],      # Конечная точка
            lineColorRGB=[1, 0, 0],             # Цвет (R, G, B) от 0 до 1
            lineWidth=2,                        # Толщина линии
            replaceItemUniqueId=self.id  
        )

def cam_init():
    cam_dist_slider = p.addUserDebugParameter(
        paramName="cameraDistance",
        rangeMin=1,
        rangeMax=15,
        startValue=8
    )
    cam_yaw_slider = p.addUserDebugParameter(
        paramName="cameraYaw",
        rangeMin=-90,
        rangeMax=90,
        startValue=50
    )
    cam_pitch_slider = p.addUserDebugParameter(
        paramName="cameraPitch",
        rangeMin=-90,
        rangeMax=90,
        startValue=-30
    )
    trg_pos = [0, 3, 0]
    trg_pos_names = ["_x", "_y", "_z"]
    trg_pos_sliders = [
        p.addUserDebugParameter(
            paramName=trg_pos_names[i],
            rangeMin=-5,
            rangeMax=5,
            startValue=trg_pos[i]
        ) for i in range(len(trg_pos))
    ]

    def cam_upd():
        cameraDistance = p.readUserDebugParameter(cam_dist_slider)
        cameraYaw = p.readUserDebugParameter(cam_yaw_slider)
        cameraPitch = p.readUserDebugParameter(cam_pitch_slider)
        cameraTargetPosition = [
            p.readUserDebugParameter(trg_pos_sliders[i]) for i in range(len(trg_pos_sliders))
        ]
        p.resetDebugVisualizerCamera(
            cameraDistance=cameraDistance, 
            cameraYaw=cameraYaw,
            cameraPitch=cameraPitch, 
            cameraTargetPosition=cameraTargetPosition
        )

    return cam_upd

class BipedWalkerEnv(gym.Env):

    def __init__(self, render_mode=None, rtk=0, curriculum_stage=1):
        super().__init__()
        self.metadata = {
            "render_modes": ["human", "rgb_array"],  # Explicitly list supported modes
            "render_fps": 30,  # Optional: framerate for rendering
        }
        self.render_mode = render_mode
        # Загрузка модели
        self.physics_client = p.connect(
            p.DIRECT if self.render_mode != "human" else p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_path = os.path.join(script_dir, "URDFs")  # RLEnv/BipedWalkerEnv/URDFs
        p.setAdditionalSearchPath(search_path)
        self.robot_id = p.loadURDF("biped.urdf")

        # self.robot = p.loadURDF("biped_pybullet.urdf")
        self.plane_id = p.loadURDF("plane.urdf")
        if self.render_mode == "human":
            self.create_toggle_btn()
            self.create_joint_sliders()
            self.cam_upd = cam_init()
        self.debug()

        p.setGravity(0, 0, -9.81)

        self.rtk = rtk
        self.step_count = 0
        self.motor_joints_indices = [0, 1, 2, 3, 4, 5]
        self.curriculum_stage = curriculum_stage # 1: стоять, 2: баланс, 3: идти, 4: к цели
        self.target = Target()
        self.prev_dist_to_target = 0.0 
        self.robot_height = 1.6

        # Пространство действий (моменты для 6 суставов)
        self.action_space = gym.spaces.Box(
            low=np.array([-1., -1., -1., -1., -1., -1.], dtype=np.float32), 
            high=np.array([1., 0., 1., 1., 0., 1.], dtype=np.float32), 
            dtype=np.float32
        )
        obs_dim = len(self.motor_joints_indices) * 2 + 4 # 6 pos + 6 vel + 4 quat
        if self.curriculum_stage == 4:
            obs_dim += 2 # Добавляем относительные угол отклонения от цели (-π, +π) и расстояние
        # Пространство состояний (углы, скорости, положение корпуса и т.д.)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        # --- Начальная поза в зависимости от этапа ---
        start_pos = [0, 0, self.robot_height+0.01]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])


        if self.curriculum_stage == 2: # Этап 2: Баланс с возмущением
             # Добавляем небольшое случайное отклонение
            roll = np.random.uniform(-0.2, 0.2)
            pitch = np.random.uniform(-0.2, 0.2)
            start_orn = p.getQuaternionFromEuler([roll, pitch, 0])

        # Сброс робота в начальное положение
        p.resetBasePositionAndOrientation(self.robot_id, start_pos, start_orn)
        for joint in range(p.getNumJoints(self.robot_id)):
            p.resetJointState(self.robot_id, joint, 0)

        # --- Цель для этапа 4 ---
        if self.curriculum_stage == 4:
            self.target.rng = self.np_random
            self.target.randomize_target_pos(angle1=-np.pi/6, angle2=np.pi/6)

            # Получаем начальную позицию робота
            torso_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            # Вычисляем начальное расстояние и сохраняем его как "предыдущее"
            self.prev_dist_to_target = np.linalg.norm(self.target.pos[:2] - np.array(torso_pos[:2]))
            
        return self._get_obs(), {}


    def step(self, action):
        time_before = time.time()
        # for i in range(p.getNumJoints(self.robot)):
        #     p.setJointMotorControl2(
        #         self.robot,
        #         i,
        #         controlMode=p.TORQUE_CONTROL,
        #         force=action[i] * 250
        #     )
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.motor_joints_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=action * np.pi / 2, # Масштабируем действие
            forces=[150] * len(self.motor_joints_indices) # Максимальное усилие
        )

        if self.render_mode == "human":
            if hasattr(self, 'joint_sliders'):
                self.update_joints_from_sliders()

        p.stepSimulation()

        obs = self._get_obs()

        reward = self._get_reward()
        falled = self._check_fall()
        reward += -50 if falled else 0
        done = falled

        self.step_count += 1
        sim_drt = time.time() - time_before
        truncated = False # Здесь можно добавить условие по времени
        info = {
            "step_count": self.step_count,
            "sim_drt": sim_drt
        }

        
        if self.render_mode == "human":
            self.cam_upd()
            if self.rtk > 0:
                time_to_wait = (self.rtk / 240.0) - sim_drt
                if time_to_wait > 0:
                    time.sleep(time_to_wait)

        return obs, reward, done, truncated, info

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot_id, range(6))

        angles = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot_id)
        r, deviation_angle = self.robot_polar_target_pos(torso_pos, torso_orn)

        obs = np.concatenate([
            angles,             # x6
            velocities,         # x6
            torso_orn,          # x4
        ], dtype=np.float32)

        # Для этапа 4 добавляем информацию о цели
        if self.curriculum_stage == 4:
            obs = np.concatenate([
                obs,
                [r, deviation_angle],  # x2
            ], dtype=np.float32)

        return obs

    def _get_reward(self):
        if self.curriculum_stage  == 1 or self.curriculum_stage  == 2:
            reward = self._calculate_standing_reward()
            return reward
        elif self.curriculum_stage  == 3:  # Ходьба
            reward = self._calculate_walking_reward()
            return reward
        elif self.curriculum_stage  == 4:  # Повороты
            reward = self._calculate_target_reward()
            return reward
        
    def _calculate_standing_reward(self):
        # Этап 1 и 2: Стоять и балансировать
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # 1. Бонус за выживание (быть "живым")
        alive_bonus = 1.0
        
        # 2. Бонус за вертикальное положение
        # Вектор "вверх" для торса. В идеале [0,0,1]
        rot_matrix = p.getMatrixFromQuaternion(torso_orn)
        up_vector = np.array(rot_matrix[6:]) # Третий столбец матрицы вращения
        upright_reward = up_vector[2] # Чем ближе Z к 1, тем лучше
        
        # 3. Штраф за движение (мы хотим стоять на месте)
        torso_vel, torso_ang_vel = p.getBaseVelocity(self.robot_id)
        velocity_penalty = -(np.linalg.norm(torso_vel) + np.linalg.norm(torso_ang_vel))
        
        # 4. Штраф за энергопотребление/усилие
        joint_states = p.getJointStates(self.robot_id, self.motor_joints_indices)
        joint_torques = [state[3] for state in joint_states]
        effort_penalty = -np.linalg.norm(joint_torques)

        # --- НОВЫЙ КОМПОНЕНТ: Награда за высоту ---
        target_height = self.robot_height  # <<< ЗАДАЙТЕ ЦЕЛЕВУЮ ВЫСОТУ ДЛЯ ВАШЕГО РОБОТА
        current_height = torso_pos[2]
        
        # Мы хотим, чтобы отклонение от target_height было минимальным.
        # Используем экспоненту, чтобы штраф был маленьким около цели и быстро рос при удалении.
        # Коэффициент k определяет "строгость" штрафа.
        height_diff_penalty = -(target_height - current_height)**2
        # Эта функция дает 1.0, когда current_height == target_height, и быстро падает до 0.

        return np.sum([
            3.0     * alive_bonus,
            1.0     * upright_reward,
            0.3     * velocity_penalty,
            0.006   * effort_penalty,
            2.0     * height_diff_penalty
        ], dtype=np.float32)
    
    def _calculate_walking_reward(self):
        # Этап 3: Идти прямо
        
        # Награда за стояние - хорошая основа
        base_reward = self._calculate_standing_reward()
        
        # 1. Главный бонус - за скорость вперед (по оси Y)
        torso_vel, _ = p.getBaseVelocity(self.robot_id)
        forward_velocity = torso_vel[1]
        
        # 2. Штраф за отклонение от прямой (движение по оси X)
        sideways_penalty = - abs(torso_vel[0])
        
        return np.sum([
            1   * base_reward,
            2.0 * forward_velocity,
            1.0 * sideways_penalty
        ], dtype=np.float32)

    def _calculate_target_reward(self):
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot_id)
        # Этап 4: Идти к цели
        
        # Награда за стояние все еще полезна
        base_reward = self._calculate_standing_reward()
        
        # 1. Вычисляем ТЕКУЩЕЕ расстояние до цели
        current_dist_to_target = np.linalg.norm(self.target.pos[:2] - np.array(torso_pos[:2]))
        
        # 2. Вычисляем награду за прогресс ("Горячо-Холодно")
        # Мы умножаем на коэффициент (например, 1000), чтобы сделать этот сигнал значимым
        progress_reward = (self.prev_dist_to_target - current_dist_to_target) * 1000 
        
        # 3. КРИТИЧЕСКИ ВАЖНО: Обновляем "предыдущее" расстояние для СЛЕДУЮЩЕГО шага
        self.prev_dist_to_target = current_dist_to_target
        
        # 2. Большой бонус за достижение цели
        reach_bonus = 0
        if current_dist_to_target < 0.5: # Порог достижения цели
            print("Цель достигнута!")
            # Можно сделать terminated = True здесь, но это сложнее
            # Проще дать большой бонус и сбросить цель в reset
            reach_bonus = 50
            
        return np.sum([
            1   * base_reward,
            3.0 * progress_reward,
            1.0 * reach_bonus
        ], dtype=np.float32)

    def _reward_balance_OUTDATED(self):
        # N = 1000  # Общее число шагов
        # total_reward = 1000  # Желаемая сумма наград
        # k = 0.001  # Коэффициент затухания (можно менять)
        # t = self.step_count

        # # Вычисляем начальную награду r0
        # r0 = total_reward * (1 - np.exp(-k)) / (1 - np.exp(-k * N))
        # stability_reward = r0 * np.exp(-k * t)
        stability_reward = .1

        # Штраф за наклон
        tilt_pen = self.tilt_penalty(max_allowed_tilt=20)

        return stability_reward, tilt_pen

    def _reward_walk_OUTDATED(self):
        velocity_reward = self.get_speed_projection([0, 1, 0]) # Скорость по Y
        flight_penalty = self.calc_flight_penalty()
        return velocity_reward, flight_penalty

    def _reward_turn_OUTDATED(self):

        "доделать"

        # Награда за поворот + движение
        angle_error = abs(self.robot_yaw - self.target_angle)
        return -1000
        return -angle_error + 0.1 * self.robot_velocity_x
    
    def _calculate_reward_OUTDATED(self, action):
        done = False
        # Получаем текущую позицию и ориентацию, и скорость
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot_id)
        torso_vel, _ = p.getBaseVelocity(self.robot_id)
        
        # 1. Награда за приближение к цели (основная компонента)
        target_pos = np.array(self.target.pos[:2], dtype=np.float32)  # Координаты X,Y цели
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
        torso_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        return torso_pos[2] < 0.5

    def tilt_penalty_OUTDATED(self, max_allowed_tilt=15):
        """
        Возвращает штраф (отрицательное число) за наклон корпуса.
        max_allowed_tilt - максимальный допустимый наклон в градусах.
        """
        pitch, roll = self.get_body_tilt_angles()
        max_tilt = max(abs(pitch), abs(roll))
        
        if max_tilt <= max_allowed_tilt:
            return 0  # Нет штрафа
        else:
            # Квадратичный штраф за превышение угла
            penalty = -0.01 * (max_tilt - max_allowed_tilt)**2
            return penalty

    def get_body_tilt_angles_OUTDATED(self):
        """
        Возвращает углы наклона корпуса (тангаж и крен) в градусах.
        Тангаж (pitch) - наклон вперед/назад, крен (roll) - наклон вбок.
        """
        _, orientation = p.getBasePositionAndOrientation(self.robot_id)
        quat = orientation
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        
        # Вычисляем тангаж (pitch) и крен (roll) в радианах
        pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        
        # Переводим в градусы
        return np.degrees(pitch), np.degrees(roll)

    def get_speed_projection(self, direction_vector):
        """
        Возвращает проекцию скорости робота на заданный вектор направления.
        
        Параметры:
            direction_vector (np.array): Вектор направления [x, y, z] (любой длины).
        
        Возвращает:
            float: Проекция скорости в м/с (скаляр).
        """
        # Получаем линейную скорость корпуса в глобальных координатах
        linear_velocity, _ = p.getBaseVelocity(self.robot_id)
        v = np.array(linear_velocity, dtype=np.float16)
        
        # Нормируем вектор направления
        d = np.array(direction_vector, dtype=np.float16)
        d_norm = d / np.linalg.norm(d) if np.linalg.norm(d) > 0 else np.zeros(3)
        
        # Вычисляем проекцию скорости (скалярное произведение)
        return np.dot(v, d_norm, dtype=np.float16)

    def create_toggle_btn(self):
        self.toggle_btn = p.addUserDebugParameter("Toggle controls", 1, 0, 0)

    def create_joint_sliders(self):
        self.joint_sliders = {}
        num_joints = p.getNumJoints(self.robot_id)
        
        for joint_idx in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
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
                    bodyUniqueId=self.robot_id,
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
                bodyA=self.robot_id,
                linkIndexA=foot_link_id,
                bodyB=self.plane_id  # ID плоскости (обычно 0)
            )
            if (len(contact_info)):
                foots_on_ground += 1 
            
        is_in_air = foots_on_ground == 0
        
        # Штрафуем только если робот поднялся слишком высоко
        torso_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        if is_in_air and torso_pos[2] > 2.0:  # Порог высоты (настройте под ваш робот)
            flight_penalty = -2.0  # Жёсткий штраф за полёт
        elif is_in_air:
            flight_penalty = -0.6  # Мягкий штраф за отрыв
        
        return flight_penalty
    
    def exp_reward(self, value: np.float32, max_reward=10.0, scale=0.001):
        return np.min([max_reward, np.exp(scale * value) - 1])
    
    def robot_polar_target_pos(self, torso_pos, torso_orn, dbg=True):
        # Вычисляем угол отклонения
        rotation_matrix = np.array(p.getMatrixFromQuaternion(torso_orn), dtype=np.float32).reshape(3, 3)
        forward_vector = rotation_matrix[:, 1]
        target_vector = np.array(self.target.pos[:2]) - np.array(torso_pos[:2],dtype=np.float32)
        angle = self.angle_between_vectors(forward_vector[:2], target_vector[:2])
        r = np.linalg.norm(torso_pos[:2]-self.target.pos[:2])

        if dbg:
            # Вектор "вперёд" (красный)
            fv = p.addUserDebugLine(
                torso_pos, 
                [torso_pos[0] + forward_vector[0], 
                  torso_pos[1] + forward_vector[1], 
                  torso_pos[2]], 
                [1, 0, 0], 
                2, 
                lifeTime=1./240.
            )

            # Вектор к цели (зелёный)
            tv = p.addUserDebugLine(
                torso_pos, 
                [self.target.pos[0], self.target.pos[1], torso_pos[2]], 
                [0, 1, 0], 
                2, 
                lifeTime=1./240.
            )

        return  r, angle
    
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
            for i in range(p.getNumJoints(self.robot_id)):
                info = p.getJointInfo(self.robot_id, i)
                print(f"Joint ID: {i}, Name: {info[1].decode('utf-8')}, Link: {info[12].decode('utf-8')}")
                

            times =+ 1

    def set_phase(self, phase):
        self.phase = phase

    def render(self):
        # Этот метод вызывается только если render_mode='rgb_array'
        if self.render_mode != "rgb_array":
            return None

        width, height = 640, 480
        
        # Получаем позицию торса, чтобы камера следовала за ним
        torso_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        # Настройки камеры
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=torso_pos, # Камера смотрит на торс робота
            distance=5.0,                  # Расстояние от камеры до робота
            yaw=90,                        # Угол поворота камеры вокруг вертикальной оси
            pitch=-20,                     # Угол наклона камеры
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, 
            aspect=float(width)/height,
            nearVal=0.1, 
            farVal=100.0
        )
        
        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
        # 1. Получаем все нужные данные из PyBullet, включая width, height и плоский список пикселей
        img_width, img_height, rgb_pixels, _, _ = p.getCameraImage(
            width, height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # 2. Преобразуем плоский список пикселей в массив NumPy
        rgb_array = np.array(rgb_pixels, dtype=np.uint8)
        
        # 3. Меняем форму массива на (height, width, 4), так как PyBullet возвращает RGBA
        rgb_array = rgb_array.reshape((img_height, img_width, 4))
        
        # 4. Теперь, когда у нас есть правильный массив, мы можем убрать альфа-канал
        rgb_array_rgb = rgb_array[:, :, :3]
        
        return rgb_array_rgb
    
    def close(self):
        p.disconnect()