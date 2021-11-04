"""
This file implement PPO algorithm.

You need to implement `compute_loss` function.

-----

*2021-2022 1st term, IERG 5350: Reinforcement Learning. Department of Information Engineering,
The Chinese University of Hong Kong. Course Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.*
"""
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

current_dir = osp.join(osp.abspath(osp.dirname(__file__)))
sys.path.append(current_dir)
sys.path.append(osp.dirname(current_dir))
print(current_dir)

from base_trainer import BaseTrainer
from buffer import PPORolloutStorage


class PPOConfig:
    """Not like previous assignment where we use a dict as config, here we
    build a class to represent config."""

    def __init__(self):
        # Common
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.save_freq = 10
        self.log_freq = 1
        self.num_envs = 1

        # Sample
        self.num_steps = 1000  # num_steps * num_envs = sample_batch_size

        # Learning
        self.gamma = 0.99
        self.lr = 5e-5
        self.grad_norm_max = 10.0
        self.entropy_loss_weight = 0.0
        self.ppo_epoch = 10
        self.mini_batch_size = 256
        self.ppo_clip_param = 0.2
        self.USE_GAE = True
        self.gae_lambda = 0.95
        self.value_loss_weight = 1.0


ppo_config = PPOConfig()


class PPOTrainer(BaseTrainer):
    def __init__(self, env, config):
        super(PPOTrainer, self).__init__(env, config)

        # There configs are only used in PPO
        self.num_sgd_steps = config.ppo_epoch
        self.mini_batch_size = config.mini_batch_size
        self.clip_param = config.ppo_clip_param

    def setup_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def setup_rollouts(self):
        act_dim = 1 if self.discrete else self.num_actions
        self.rollouts = PPORolloutStorage(
            self.num_steps, self.num_envs, act_dim, self.num_feats[0], self.device, self.discrete,
            self.config.USE_GAE, self.config.gae_lambda
        )

    def compute_loss(self, sample):
        """Compute the loss of PPO"""
        observations_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
        old_action_log_probs_batch, adv_targ = sample

        assert old_action_log_probs_batch.shape == (self.mini_batch_size, 1)
        assert adv_targ.shape == (self.mini_batch_size, 1)
        assert return_batch.shape == (self.mini_batch_size, 1)

        values, action_log_probs, dist_entropy = self.evaluate_actions(observations_batch, actions_batch)

        assert values.shape == (self.mini_batch_size, 1)
        assert action_log_probs.shape == (self.mini_batch_size, 1)
        assert values.requires_grad
        assert action_log_probs.requires_grad
        assert dist_entropy.requires_grad

        # [TODO] Implement policy loss
        policy_loss = None
        ratio = None  # The importance sampling factor, the ratio of new policy prob over old policy prob
        pass
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_loss = -torch.min(surr1, surr2)
        policy_loss_mean = policy_loss.mean()

        # [TODO] Implement value loss
        # value_loss = None
        # pass
        # value_loss = F.mse_loss(input=values, target=return_batch)
        # ===========================================================
        vf_clip_param = 10.0
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-vf_clip_param, vf_clip_param)
        value_losses = (values - return_batch).pow(2)
        value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        value_loss_mean = value_loss.mean()

        # else:
        # value_loss = 0.5 * (return_batch - values).pow(2).mean()

        # This is the total loss
        loss = policy_loss + self.config.value_loss_weight * value_loss - self.config.entropy_loss_weight * dist_entropy
        loss = loss.mean()

        return loss, policy_loss_mean, value_loss_mean, torch.mean(dist_entropy), torch.mean(ratio)

    def update(self, rollout):
        # Get the normalized advantages
        advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
        adv_mean = advantages.mean().item()
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-4)

        value_loss_epoch = []
        policy_loss_epoch = []
        dist_entropy_epoch = []
        total_loss_epoch = []
        norm_epoch = []
        ratio_epoch = []

        assert advantages.shape[0] * advantages.shape[1] >= self.mini_batch_size, "Number of sampled steps should more than mini batch size."

        # Train for num_sgd_steps iterations (compared to A2C which only
        # train one iteration)
        for e in range(self.num_sgd_steps):
            data_generator = rollout.feed_forward_generator(advantages, self.mini_batch_size)

            for sample in data_generator:
                total_loss, policy_loss, value_loss, dist_entropy, ratio = self.compute_loss(sample)
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.config.grad_norm_max:
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_max)
                    norm = norm.item()
                else:
                    norm = 0.0
                self.optimizer.step()

                value_loss_epoch.append(value_loss.item())
                policy_loss_epoch.append(policy_loss.item())
                total_loss_epoch.append(total_loss.item())
                dist_entropy_epoch.append(dist_entropy.item())
                norm_epoch.append(norm)
                ratio_epoch.append(ratio.item())

        return np.mean(policy_loss_epoch), np.mean(value_loss_epoch), np.mean(dist_entropy_epoch), \
               np.mean(total_loss_epoch), np.mean(norm_epoch), adv_mean, np.mean(ratio_epoch)


if __name__ == '__main__':
    env_name = "MetaDrive-Tut-Hard-v0"
    from utils import register_metadrive as r
    import gym

    r()
    e = gym.make(env_name)
    test_trainer = PPOTrainer(e, ppo_config)
    test_trainer.load_w("../MetaDriveEasy/PPO", "final")
