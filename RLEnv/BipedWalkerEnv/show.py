import time
import gymnasium as gym
import os
import RegEnvs
import pybullet as p

env_name = "BipedWalkerCustom-v1"

script_dir = os.path.dirname(os.path.abspath(__file__))  # RLEnv/BipedWalkerEnv
models_dir = os.path.join(script_dir, "models")  # RLEnv/BipedWalkerEnv/models

# env = gym.make("BipedWalker-v0", render_mode="human")
env = gym.make(env_name, render_mode="human")
obs, _ = env.reset()

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

cam_upd = cam_init()
while (1):
    cam_upd()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        pass

    # time.sleep(.01)