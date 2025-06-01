import gymnasium as gym


gym.register(  
    id="BipedWalkerCustom-v0",  
    entry_point="BipedWalkerEnv:BipedWalkerEnv",
    max_episode_steps=1000,  
)

gym.register(  
    id="BipedWalkerCustom-v1",  
    entry_point="BipedWalkerEnv-v1:BipedWalkerEnv",
    max_episode_steps=1000,  
)