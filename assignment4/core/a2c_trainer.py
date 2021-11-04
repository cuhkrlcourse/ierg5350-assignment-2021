"""
This file implement A2C algorithm.

You need to implement `update` and `compute_loss` functions.

-----

*2021-2022 1st term, IERG 5350: Reinforcement Learning. Department of Information Engineering,
The Chinese University of Hong Kong. Course Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.*
"""
import torch

from .base_trainer import BaseTrainer
from .buffer import A2CRolloutStorage


class A2CConfig(object):
    def __init__(self):
        # Common
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_freq = 50
        self.log_freq = 5
        self.num_envs = 1

        # Sample
        self.num_steps = 500  # num_steps * num_envs = sample_batch_size

        # Learning
        self.gamma = 0.99
        self.lr = 7e-4
        self.grad_norm_max = 0.5
        self.entropy_loss_weight = 0.01
        self.value_loss_weight = 0.5


a2c_config = A2CConfig()


class A2CTrainer(BaseTrainer):
    def __init__(self, env, config, _test=False):
        super(A2CTrainer, self).__init__(env, config, _test=_test)

    def setup_optimizer(self):
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, eps=1e-5)

    def setup_rollouts(self):
        self.rollouts = A2CRolloutStorage(
            self.num_steps, self.num_envs, self.num_actions, self.num_feats[0], self.device, discrete=self.discrete
        )

    def compute_loss(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy = self.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, self.num_actions))

        # Note that since we are using stacked environments, the tensors are in shape
        # [num_steps, num_envs, num_features].
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # [TODO] Get the unnormalized advantages
        advantages = None
        pass
        advantages = rollouts.returns[:-1] - values

        advantages_mean = advantages.mean().item()  # Used to record statistics

        # [TODO] Get the value loss
        value_loss = None
        pass
        value_loss = advantages.pow(2).mean()

        # [TODO] Get the policy loss
        policy_loss = None
        pass
        policy_loss = -(advantages.detach() * action_log_probs).sum()


        # print(
        #     "Log Prob Mean: ", action_log_probs.mean().item(),
        #     ". Adv Mean: ", advantages.mean().item(),
        #     ". Loss: ", policy_loss.item(),
        #     ". Return: ", rollouts.returns[:-1].mean().item(),
        #     ". Value: ", values.mean().item()
        # )

        # Get the total loss
        loss = policy_loss + self.value_loss_weight * value_loss - \
               self.entropy_loss_weight * dist_entropy

        return loss, policy_loss, value_loss, dist_entropy, advantages_mean

    def update(self, rollout):
        total_loss, action_loss, value_loss, dist_entropy, adv = self.compute_loss(rollout)
        # [TODO] Step self.optimizer by computing the gradient of total loss
        # Hint: remember to clip the gradient to self.grad_norm_max and set the gradient norm to variable norm.
        norm = None
        pass
        self.optimizer.zero_grad()
        total_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_max)
        self.optimizer.step()

        return action_loss.item(), value_loss.item(), dist_entropy.item(), \
               total_loss.item(), norm.item(), adv, 0.0
