from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CurriculumCallback(BaseCallback):
    """
    Использование:
    ```
        env = make_vec_env("BipedalEnv-v0", n_envs=4)
        model = PPO("MlpPolicy", env, verbose=1)
        callback = CurriculumCallback(env, reward_threshold=500)
        model.learn(total_timesteps=1_000_000, callback=callback)
    ```
    """
    def __init__(self, env, reward_threshold=80, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.reward_threshold = reward_threshold
        self.phase = 0  # 0=balance, 1=balance_hard, 2=walk, 3=turn

    def _on_step(self) -> bool:
        # Проверяем среднюю награду каждые 10 шагов
        if self.n_calls % 10 == 0:
            # Extract rewards from all episodes in the buffer
            rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer if "r" in ep_info]
            if len(rewards) > 0:
                mean_reward = np.mean(rewards[-100:])  # Last 100 episodes
                if mean_reward >= self.reward_threshold and self.phase == 0:
                    self.phase = 1
                    self.env.env_method("set_phase", 1)  # Переключаем среду на усложненный режим ходьбы
                    print("Переход к этапу ходьбы!")
                elif mean_reward >= -60 and self.phase == 1:
                    self.phase = 2
                    self.env.env_method("set_phase", 2)  # Переключаем среду на режим ходьбы
                    print("Переход к этапу поворотов!")
        return True