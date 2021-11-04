"""
Example script to visualize your trained agent.
This file might be useful in debugging.
"""
import gym
from load_agents import PolicyAPI

env_name = "MetaDrive-Tut-10Env-v0"
policy = PolicyAPI(
    env_name,  # In order to get the observation shape
    num_envs=1,
    log_dir="1104_results/3/metadrive_10_env_20k/PPO",
    suffix="iter50"
)

comp_env = gym.make(env_name, config={'use_render': True})
obs = comp_env.reset()
ep_rew = 0.0
for i in range(1000):
    frame = comp_env.render()
    act = policy(obs)
    print("Current step: {}, Action: {}".format(i, act))
    obs, rew, term, _ = comp_env.step(act)
    ep_rew += rew
    if term:
        print("Episode reward: ", ep_rew)
        break
comp_env.close()
