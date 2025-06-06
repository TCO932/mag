import os
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.callbacks import BaseCallback

import RegEnvs
import BipedCurriculumCallback as cb 


env_name = "BipedWalkerCustom-v1"
model_name = "BipedWalkerCustom-v4"
alg = "SAC"
total_timesteps = 2_000_000
n_envs = 4

script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/BipedWalkerEnv
models_dir = os.path.join(script_dir, "models", "Curriculum", alg)  # RLEnv/BipedWalkerEnv/models/PPO
os.makedirs(models_dir, exist_ok=True)
tb_logs_path = os.path.join(script_dir, "logs", "Curriculum", model_name, alg)

# Separate evaluation env
eval_env = gym.make(env_name)
# eval_env.env_method("set_phase", 0)

# Stop training when the model reaches the reward threshold
# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=90, verbose=1)
# eval_callback = EvalCallback(
#     eval_env, 
#     callback_on_new_best=callback_on_best,
#     best_model_save_path=models_dir+"/best/",
#     verbose=1
# )

# model = SAC(
#     "MlpPolicy",
#     eval_env,
#     verbose=1,
#     # n_steps=2048,           # Увеличить для более стабильных updates
#     # batch_size=64,          # Можно попробовать увеличить до 128-256
#     # learning_rate=3e-4,   # Стандартное значение для PPO
#     # ent_coef=0.05,           # Увеличить для большего исследования
#     # gamma=0.99,
#     # gae_lambda=0.95,
#     # max_grad_norm=0.5,
#     # clip_range=0.1,         # Уменьшить для более консервативных обновлений
#     # n_epochs=10,
#     tensorboard_log=tb_logs_path,
# )
# for i in range(3):
#     eval_env.env_method("set_phase", i)
#     # Almost infinite number of timesteps, but the training will stop
#     # early as soon as the reward threshold is reached
#     model.learn(
#         total_timesteps=int(1e10),
#         progress_bar=True,
#         reset_num_timesteps=False,  # Сохраняем предыдущий прогресс
#         tb_log_name=tb_logs_path, 
#         callback=eval_callback
#     )
#     model.save(models_dir+"/"+model_name+f"phase_{i}")

# env = make_vec_env("BipedalEnv-v0", n_envs=4)
# model = PPO("MlpPolicy", env, verbose=1)
env = gym.make(env_name)
model = SAC(
    "MlpPolicy",
    eval_env,
    verbose=1,
    # n_steps=2048,           # Увеличить для более стабильных updates
    # batch_size=64,          # Можно попробовать увеличить до 128-256
    # learning_rate=3e-4,   # Стандартное значение для PPO
    # ent_coef=0.05,           # Увеличить для большего исследования
    # gamma=0.99,
    # gae_lambda=0.95,
    # max_grad_norm=0.5,
    # clip_range=0.1,         # Уменьшить для более консервативных обновлений
    # n_epochs=10,
    tensorboard_log=tb_logs_path,
)
callback = cb.CurriculumCallback(env, reward_threshold=500)
model.learn(
    total_timesteps=1_000_000, 
    progress_bar=True,
    callback=callback
)


# class CurriculumCallback(BaseCallback):
#     def __init__(self, env, reward_threshold=500, verbose=0):
#         super().__init__(verbose)
#         self.env = env
#         self.reward_threshold = reward_threshold
#         self.phase = 0  # 0=balance, 1=walk, 2=turn

#     def _on_step(self) -> bool:
#         # Проверяем среднюю награду каждые 10 шагов
#         if self.n_calls % 10 == 0:
#             mean_reward = np.mean(self.model.ep_info_buffer["r"][-100:])
#             if mean_reward >= self.reward_threshold and self.phase == 0:
#                 self.phase = 1
#                 self.env.env_method("set_phase", 1)  # Переключаем среду на режим ходьбы
#                 print("Переход к этапу ходьбы!")
#             elif mean_reward >= 800 and self.phase == 1:
#                 self.phase = 2
#                 self.env.env_method("set_phase", 2)  # Переключаем на повороты
#                 print("Переход к этапу поворотов!")
#         return True

# # Использование:
# env = make_vec_env("BipedalEnv-v0", n_envs=4)
# model = PPO("MlpPolicy", env, verbose=1)
# callback = CurriculumCallback(env, reward_threshold=500)
# model.learn(total_timesteps=1_000_000, callback=callback)