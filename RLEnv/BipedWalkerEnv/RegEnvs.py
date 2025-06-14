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

gym.register(  
    id="BipedWalkerCustom-v2-stage-1",  
    entry_point="BipedWalkerEnv-v1:BipedWalkerEnv",
    max_episode_steps=1000,  
    kwargs={"curriculum_stage":1}
)
gym.register(  
    id="BipedWalkerCustom-v2-stage-2",  
    entry_point="BipedWalkerEnv-v1:BipedWalkerEnv",
    max_episode_steps=1000,  
    kwargs={"curriculum_stage":2}
)
gym.register(  
    id="BipedWalkerCustom-v2-stage-3",  
    entry_point="BipedWalkerEnv-v1:BipedWalkerEnv",
    max_episode_steps=3000,
    kwargs={"curriculum_stage":3}
)
gym.register(  
    id="BipedWalkerCustom-v2-stage-4",  
    entry_point="BipedWalkerEnv-v1:BipedWalkerEnv",
    max_episode_steps=3000,  
    kwargs={"curriculum_stage":4}
)