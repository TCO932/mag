import gymnasium as gym
from gymnasium.envs.registration import registry
import main

# Печатаем все зарегистрированные среды
for env_id, env_spec in registry.items():
    print(f"ID: {env_id}")
    print(f"Entry Point: {env_spec.entry_point}")
    print("------")