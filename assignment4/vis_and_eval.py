"""
Example script to visualize or evaluate your trained agent.
This file might be useful in debugging.
"""
import gym
from load_agents import PolicyAPI
from collections import deque

env_name = "MetaDrive-Tut-10Env-v0"
policy = PolicyAPI(
    env_name,  # In order to get the observation shape
    num_envs=1,
    log_dir="1104_results/3/metadrive_10_env_20k/PPO",
    suffix="iter50"
)
num_episodes = 10

# Turn use_render to False if you are trying to evaluate agents.
comp_env = gym.make(env_name, config={'use_render': True})
obs = comp_env.reset()

ep_rew = 0.0
reward_recorder = deque(maxlen=10)
success_recorder = deque(maxlen=10)
episode_count = 0

while True:
    frame = comp_env.render()
    act = policy(obs)
    # print("Current step: {}, Action: {}".format(i, act))
    obs, rew, term, info = comp_env.step(act)
    ep_rew += rew
    if term:
        print("Episode reward: ", ep_rew)
        success = info["arrive_dest"]
        success_recorder.append(float(success))
        reward_recorder.append(float(ep_rew))
        ep_rew = 0
        episode_count += 1
        obs = comp_env.reset()
        if episode_count >= num_episodes:
            break
comp_env.close()
