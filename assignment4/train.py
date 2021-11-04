"""
This file implements the train scripts for both A2C and PPO

You need to implement all TODOs in this script.

-----
*2021-2022 1st term, IERG 5350: Reinforcement Learning. Department of Information Engineering,
The Chinese University of Hong Kong. Course Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.*
"""
import argparse
from collections import deque

import gym
import numpy as np
import torch
from core.a2c_trainer import A2CTrainer, a2c_config
from core.envs import make_envs
from core.ppo_trainer import PPOTrainer, ppo_config
from core.utils import verify_log_dir, pretty_print, Timer, summary, save_progress, step_envs

gym.logger.set_level(40)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--algo",
    default="",
    type=str,
    choices=["PPO", "A2C"],
    help="(Required) The algorithm you want to run. Must in [PPO, A2C]."
)
parser.add_argument(
    "--log-dir",
    default="data/",
    type=str,
    help="The path of directory that you want to store the data to. "
         "Default: ./data/"
)
parser.add_argument(
    "--num-envs",
    default=10,
    type=int,
    help="The number of parallel environments. Default: 10"
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="The random seed. Default: 0"
)
parser.add_argument(
    "--max-steps",
    "-N",
    default=1e7,
    type=float,
    help="The random seed. Default: 1e7"
)
parser.add_argument(
    "--num-steps",
    default=2000,
    type=int,
)
parser.add_argument(
    "--env-id",
    default="MetaDrive-Tut-Easy-v0",
    type=str,
)
parser.add_argument(
    "--synchronous",
    action="store_true"
)
parser.add_argument(
    "-lr",
    default=-1,
    type=float,
    help="Learning rate. Default: None"
)
parser.add_argument(
    "--gae-lambda",
    default=0.95,
    type=float,
    help="GAE coefficient. Default: 0.95"
)
parser.add_argument(
    "--num-epoch",
    default=10,
    type=int,
)
args = parser.parse_args()

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(obs):
    obs = torch.from_numpy(obs.astype(np.float32)).to(default_device)
    return obs


def train(args):
    # Verify algorithm and config
    algo = args.algo
    if algo == "PPO":
        config = ppo_config
    elif algo == "A2C":
        config = a2c_config
    else:
        raise ValueError("args.algo must in [PPO, A2C]")
    config.num_envs = args.num_envs
    config.num_steps = args.num_steps
    config.gae_lambda = args.gae_lambda
    config.ppo_epoch = args.num_epoch
    if args.lr != -1:
        config.lr = args.lr

    # Seed the environments and setup torch
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)

    # Clean log directory
    log_dir = verify_log_dir(args.log_dir, algo)

    # Create vectorized environments
    num_envs = args.num_envs
    env_id = args.env_id
    envs = make_envs(
        env_id=env_id,
        log_dir=log_dir,
        num_envs=num_envs,
        asynchronous=not args.synchronous,
    )

    # Setup trainer
    if algo == "PPO":
        trainer = PPOTrainer(envs, config)
    else:
        trainer = A2CTrainer(envs, config)

    # Setup some stats helpers
    episode_rewards = np.zeros([num_envs, 1], dtype=np.float)
    total_episodes = total_steps = iteration = 0
    reward_recorder = deque(maxlen=100)
    success_recorder = deque(maxlen=100)
    sample_timer = Timer()
    process_timer = Timer()
    update_timer = Timer()
    total_timer = Timer()
    progress = []
    evaluate_stat = {}

    # Start training
    print("Start training!")
    obs = envs.reset()
    trainer.rollouts.observations[0].copy_(to_tensor(obs))
    while True:  # Break when total_steps exceeds maximum value
        # ===== Sample Data =====
        with sample_timer:
            for index in range(config.num_steps):
                # Get action
                # [TODO] Get the action, values, and action log prob. from trainer, all are tensors.
                # Hint:
                #   1. Remember to disable gradient when collecting data here
                #   2. trainer.rollouts is a storage containing all data
                #   3. Pass current observations to compute_action
                #   4. Since we are using stacked environments, please pay attention to the shape of each type of data
                values = None
                actions = None
                action_log_prob = None
                pass
                with torch.no_grad():
                    values, actions, action_log_prob = \
                        trainer.compute_action(trainer.rollouts.observations[index])

                # actions is a torch tensor, so we need to turn it into numpy array.
                cpu_actions = actions.cpu().numpy()
                if trainer.discrete:
                    cpu_actions = cpu_actions.reshape(-1)  # flatten

                # Step the environment
                # (Check step_envs function, you need to implement it)
                obs, reward, done, info, masks, total_episodes, \
                total_steps, episode_rewards = step_envs(
                    cpu_actions, envs, episode_rewards,
                    reward_recorder, success_recorder, total_steps,
                    total_episodes, config.device
                )

                rewards = torch.from_numpy(
                    reward.astype(np.float32)).view(-1, 1).to(config.device)

                # Store samples
                trainer.rollouts.insert(to_tensor(obs), actions, action_log_prob, values, rewards, masks)

        # ===== Process Samples =====
        with process_timer:
            with torch.no_grad():
                next_value = trainer.compute_values(trainer.rollouts.observations[-1])
            trainer.rollouts.compute_returns(next_value, config.gamma)

        # ===== Update Policy =====
        with update_timer:
            policy_loss, value_loss, dist_entropy, total_loss, norm, adv, ratio = trainer.update(trainer.rollouts)
            trainer.rollouts.after_update()

        # ===== Log information =====
        if iteration % config.log_freq == 0:
            stats = dict(
                log_dir=log_dir,
                frame_per_second=int(total_steps / total_timer.now),
                training_episode_reward=summary(reward_recorder, "episode_reward"),
                success_rate=summary(success_recorder, "success_rate"),
                evaluate_stats=evaluate_stat,
                learning_stats=dict(
                    policy_loss=policy_loss,
                    entropy=dist_entropy,
                    value_loss=value_loss,
                    total_loss=total_loss,
                    grad_norm=norm,
                    adv_mean=adv,
                    ratio=ratio
                ),
                total_steps=total_steps,
                total_episodes=total_episodes,
                time_stats=dict(
                    sample_time=sample_timer.avg,
                    process_time=process_timer.avg,
                    update_time=update_timer.avg,
                    total_time=total_timer.now,
                    episode_time=sample_timer.avg + process_timer.avg +
                                 update_timer.avg
                ),
                iteration=iteration
            )

            progress.append(stats)
            pretty_print({
                "===== {} Training Iteration {} =====".format(
                    algo, iteration): stats
            })

        if iteration % config.save_freq == 0:
            trainer_path = trainer.save_w(log_dir, "iter{}".format(iteration))
            progress_path = save_progress(log_dir, progress)
            print("Saved trainer state at <{}>. Saved progress at <{}>.".format(
                trainer_path, progress_path
            ))

        if total_steps > int(args.max_steps):
            break

        iteration += 1

    trainer.save_w(log_dir, "final")
    envs.close()


if __name__ == '__main__':
    train(args)
