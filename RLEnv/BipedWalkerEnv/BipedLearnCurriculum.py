import os
import gymnasium as gym

import inquirer
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize

import RegEnvs

import torch as th
import hashlib
import json


env_name = "BipedWalkerCustom-v1"
model_name = "BipedWalkerCustom-2"
alg_name = "PPO"
algs = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
alg = algs[alg_name]
mode = "norender"
mode = "human"
stage = 4

total_timesteps = 5_000_000
n_envs = 4

if __name__ == "__main__":
    algs_names = list(algs.keys())
    choice = inquirer.prompt([
        inquirer.List("algorithm", message="Select algorithm", choices=algs_names, default=algs_names[1])
    ])
    alg_name = choice["algorithm"]
    alg = algs[alg_name]
    print(f"Using {alg_name}...")
    user_model_name = input("Enter model name: ")
    model_name = f"{model_name}_{user_model_name}"

script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/BipedWalkerEnv
models_dir = os.path.join(script_dir, "models", "Curriculum", alg_name)  # RLEnv/BipedWalkerEnv/models/PPO
os.makedirs(models_dir, exist_ok=True)

alg_params = {
    "PPO": {
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "learning_rate": 1e-4,
        "ent_coef": 0.01,
        "gamma": 0.99,
        "clip_range": 0.2,
        "max_grad_norm": 0.5,
        'policy_kwargs': dict(
            net_arch=[256, 256], # Делаем нейросеть чуть побольше
            activation_fn=th.nn.ReLU
        )
    },
    "SAC": {
        "learning_rate": 3e-4,
        "buffer_size": 1_000_000,
        "learning_starts": 20_000,
        "batch_size": 256,
        "ent_coef": 'auto',
        "gamma": 0.98,
        "tau": 0.005,
        'policy_kwargs': dict(
            net_arch=[256, 256], # Делаем нейросеть чуть побольше
            activation_fn=th.nn.ReLU
        )
    },
    "TD3": {
        "learning_rate": 3e-4,
        # "buffer_size": 1000000,
        # "batch_size": 100,
        # "gamma": 0.99,
        # "tau": 0.005,
        # "policy_delay": 2,
        # "target_policy_noise": 0.2,
        # "target_noise_clip": 0.5,
    }
}

def get_params_hash(params):
    params_str = json.dumps(
        params,
        sort_keys=True,
        default=lambda o: o.__name__ if isinstance(o, type) else str(o)  # Преобразуем типы в строки
    )
    return hashlib.md5(params_str.encode()).hexdigest()

params = alg_params.get(alg_name, {})
# params_hash = get_params_hash(params)
# model_name = f"{model_name}-{params_hash}"

stages_info = [
    {
        "env_name": "BipedWalkerCustom-v2-stage-1",
        "reward_threshold": 4000,  
    }, {
        "env_name": "BipedWalkerCustom-v2-stage-2",
        "reward_threshold": 90000,  
    }, {
        "env_name": "BipedWalkerCustom-v2-stage-3",
        "reward_threshold": 90000,  
    }, {
        "env_name": "BipedWalkerCustom-v2-stage-4",
        "reward_threshold": 90000,  
    }
]

def exec_stage(stage):
    print(f"Executing stage {stage}...")

    model_path = os.path.join(models_dir, model_name)
    tb_logs_path = os.path.join(script_dir, "logs", "Curriculum", alg_name, model_name)

    stage_info = stages_info[stage-1]
    env_name = stage_info["env_name"]
    reward_threshold = stage_info["reward_threshold"]

    if mode == "human":
        env = gym.make(env_name, render_mode = "human", rtk=0)
    else:
        env = make_vec_env(
            env_name,
            n_envs=n_envs,
        )
        env = VecNormalize(env, norm_obs=True, norm_reward=True) # Нормализуем среду
        
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    best_model_path = os.path.join(models_dir, "best", model_name)
    eval_callback = EvalCallback(
        env, 
        callback_on_new_best=callback_on_best,
        best_model_save_path=best_model_path,
        verbose=1
    )

    try:
        model = alg.load(model_path, env=env)
        print(f"Model {model_name} is loaded")
    except (ValueError, FileNotFoundError):
        print(f"Creating model {model_name}...")
        model = alg(
            "MlpPolicy",
            env,
            tensorboard_log=tb_logs_path,
            **params
        )
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            progress_bar=True,
            reset_num_timesteps=False,
            callback=eval_callback
        )
    except KeyboardInterrupt:
        print("Interrupted by user!")

    print(f"Saving model {model_name}...")
    model.save(model_path)

exec_stage(stage)