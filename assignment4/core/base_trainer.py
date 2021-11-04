"""
This file implement a base trainer class.

You should finish `compute_action` and run this file to verify the base trainer is implement correctly.

-----

*2021-2022 1st term, IERG 5350: Reinforcement Learning. Department of Information Engineering,
The Chinese University of Hong Kong. Course Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.*
"""
import os
import os.path as osp
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

current_dir = osp.join(osp.abspath(osp.dirname(__file__)))
sys.path.append(current_dir)
sys.path.append(osp.dirname(current_dir))
print(current_dir)

from envs import make_envs


class BaseTrainer:
    def __init__(self, env, config, _test=False):
        self.device = config.device
        self.config = config
        self.lr = config.lr
        self.num_envs = config.num_envs
        self.gamma = config.gamma
        self.num_steps = config.num_steps

        self.grad_norm_max = config.grad_norm_max
        self.value_loss_weight = config.value_loss_weight
        self.entropy_loss_weight = config.entropy_loss_weight

        if isinstance(env.action_space, gym.spaces.Box):
            # Continuous action space
            self.discrete = False
        else:
            self.discrete = True

        if isinstance(env.observation_space, gym.spaces.Tuple):
            num_feats = env.observation_space[0].shape
            self.num_actions = env.action_space[0].n
        else:
            num_feats = env.observation_space.shape
            if self.discrete:
                self.num_actions = env.action_space.n
            else:
                self.num_actions = env.action_space.shape[0]
        self.num_feats = num_feats  # (num channel, width, height)

        self.model = MLP(num_feats[0], self.num_actions, self.discrete)

        self.model = self.model.to(self.device)
        self.model.train()

        self.setup_optimizer()
        self.setup_rollouts()

    def setup_optimizer(self):
        raise NotImplementedError()

    def setup_rollouts(self):
        raise NotImplementedError()

    def compute_loss(self, rollouts):
        raise NotImplementedError()

    def update(self, rollout):
        raise NotImplementedError()

    def process_obs(self, obs):
        # Change to tensor, change type, add batch dimension for observation.
        if not isinstance(obs, torch.Tensor):
            obs = np.asarray(obs)
            obs = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        obs = obs.float()
        if obs.ndim == 1 or obs.ndim == 3:  # Add additional batch dimension.
            obs = obs.view(1, *obs.shape)
        return obs

    def compute_action(self, obs, deterministic=False):
        obs = self.process_obs(obs)

        # [TODO] Get the actions and the log probability of the action from the output of neural network.
        #  Hint: Use proper torch distribution to help you
        actions, action_log_probs = None, None

        if self.discrete:  # Please use categorical distribution.
            logits, values = self.model(obs)
            pass
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                actions = dist.probs.argmax(dim=1, keepdim=True)
            else:
                actions = dist.sample()
            action_log_probs = dist.log_prob(actions.view(-1))

            actions = actions.view(-1, 1)  # In discrete case only return the chosen action.

        else:  # Please use normal distribution.
            means, log_std, values = self.model(obs)
            pass
            action_std = torch.exp(log_std)
            dist = torch.distributions.Normal(means, action_std)
            if deterministic:  # Use the means as the action
                actions = means
            else:
                actions = dist.sample()
            action_log_probs = dist.log_prob(actions).sum(axis=1)

            actions = actions.view(-1, self.num_actions)

        values = values.view(-1, 1)
        action_log_probs = action_log_probs.view(-1, 1)

        return values, actions, action_log_probs

    def evaluate_actions(self, obs, act):
        """Run models to get the values, log probability and action
        distribution entropy of the action in current state"""

        obs = self.process_obs(obs)

        if self.discrete:
            assert not torch.is_floating_point(act)
            logits, values = self.model(obs)
            pass
            dist = Categorical(logits=logits)
            action_log_probs = dist.log_prob(act.view(-1)).view(-1, 1)
            dist_entropy = dist.entropy()
        else:
            assert torch.is_floating_point(act)
            means, log_std, values = self.model(obs)
            pass
            action_std = torch.exp(log_std)
            dist = torch.distributions.Normal(means, action_std)
            action_log_probs_raw = dist.log_prob(act)
            action_log_probs = action_log_probs_raw.sum(axis=-1)
            dist_entropy = dist.entropy().sum(-1)

        values = values.view(-1, 1)
        action_log_probs = action_log_probs.view(-1, 1)

        return values, action_log_probs, dist_entropy

    def compute_values(self, obs):
        """Compute the values corresponding to current policy at current
        state"""
        obs = self.process_obs(obs)
        if self.discrete:
            _, values = self.model(obs)
        else:
            _, _, values = self.model(obs)
        return values

    def save_w(self, log_dir="", suffix=""):
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, "checkpoint-{}.pkl".format(suffix))
        torch.save(dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict()
        ), save_path)
        return save_path

    def load_w(self, log_dir="", suffix=""):
        log_dir = os.path.abspath(os.path.expanduser(log_dir))
        save_path = os.path.join(log_dir, "checkpoint-{}.pkl".format(suffix))
        if os.path.isfile(save_path):
            state_dict = torch.load(
                save_path,
                torch.device('cpu') if not torch.cuda.is_available() else None
            )
            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            print("Successfully load weights from {}!".format(save_path))
            return True
        else:
            print("Failed to load weights from {}!".format(save_path))
            return False


def normc_initializer(std=1.0):
    def initializer(tensor):
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(tensor.data.pow(2).sum(1, keepdim=True))

    return initializer


class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self,
                 in_size,
                 out_size,
                 initializer=None,
                 activation_fn=True,
                 use_bias=True,
                 bias_init=0.0):
        super(SlimFC, self).__init__()
        layers = []
        # Actual Conv2D layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer:
            initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        if activation_fn:
            activation_fn = nn.ReLU
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, discrete):
        super(MLP, self).__init__()

        # Setup the log std output for continuous action space
        self.discrete = discrete
        self.use_free_logstd = True
        if discrete:
            self.actor_logstd = None
        else:
            # self.actor_logstd = nn.Parameter(torch.zeros(1, output_size))
            output_size = output_size * 2
            self.use_free_logstd = False

        hidden_size = 256
        self.policy = nn.Sequential(
            SlimFC(
                in_size=input_size,
                out_size=hidden_size,
                initializer=normc_initializer(1.0),
                activation_fn=True
            ),
            SlimFC(
                in_size=hidden_size,
                out_size=hidden_size,
                initializer=normc_initializer(1.0),
                activation_fn=True
            ),
            SlimFC(
                in_size=hidden_size,
                out_size=output_size,
                initializer=normc_initializer(0.01),  # Make the output close to zero, in the beginning!
                activation_fn=False
            )
        )

        self.value = nn.Sequential(
            SlimFC(
                in_size=input_size,
                out_size=hidden_size,
                initializer=normc_initializer(1.0),
                activation_fn=True
            ),
            SlimFC(
                in_size=hidden_size,
                out_size=hidden_size,
                initializer=normc_initializer(1.0),
                activation_fn=True
            ),
            SlimFC(
                in_size=hidden_size,
                out_size=1,
                initializer=normc_initializer(0.01),  # Make the output close to zero, in the beginning!
                activation_fn=False
            )
        )

    def forward(self, input_obs):
        logits = self.policy(input_obs)
        value = self.value(input_obs)
        if self.discrete:
            return logits, value
        else:
            if self.use_free_logstd:
                return logits, self.actor_logstd, value
            else:
                mean, log_std = torch.chunk(logits, 2, dim=-1)
                return mean, log_std, value


def test_base_trainer():
    class FakeConfig:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.num_envs = 1
            self.num_steps = 200
            self.gamma = 0.99
            self.lr = 5e-4
            self.grad_norm_max = 10.0
            self.value_loss_weight = 1.0
            self.entropy_loss_weight = 0.0

    class FakeTrainer(BaseTrainer):
        def setup_optimizer(self):
            pass

        def setup_rollouts(self):
            pass

    env = make_envs("CartPole-v0", num_envs=3)
    trainer = FakeTrainer(env, FakeConfig())
    obs = env.reset()
    # Input single observation
    values, actions, action_log_probs = trainer.compute_action(obs[0], deterministic=True)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs[0], actions)
    assert actions.shape == (1, 1), actions.shape
    assert values.shape == (1, 1), values.shape
    assert action_log_probs.shape == (1, 1), action_log_probs.shape
    assert dist_entropy.shape == ()
    assert (values == new_values).all()
    assert (action_log_probs == new_action_log_probs).all()
    assert dist_entropy.shape == ()

    # Input multiple observations
    values, actions, action_log_probs = trainer.compute_action(obs, deterministic=False)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs, actions)
    assert actions.shape == (3, 1), actions.shape
    assert values.shape == (3, 1), values.shape
    assert action_log_probs.shape == (3, 1), action_log_probs.shape
    assert dist_entropy.shape == ()
    assert (values == new_values).all()
    assert (action_log_probs == new_action_log_probs).all()
    assert dist_entropy.shape == ()

    print("Base trainer discrete case test passed!")
    env.close()

    # ===== Continuous case =====
    env = make_envs("BipedalWalker-v3", asynchronous=False, num_envs=3)
    trainer = FakeTrainer(env, FakeConfig())
    obs = env.reset()
    # Input single observation
    values, actions, action_log_probs = trainer.compute_action(obs[0], deterministic=True)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs[0], actions)
    assert env.envs[0].action_space.shape[0] == actions.shape[1]
    assert values.shape == (1, 1), values.shape
    assert action_log_probs.shape == (1, 1), action_log_probs.shape
    assert dist_entropy.shape == ()
    assert (values == new_values).all()
    assert (action_log_probs == new_action_log_probs).all()
    assert dist_entropy.shape == ()

    # Input multiple observations
    values, actions, action_log_probs = trainer.compute_action(obs, deterministic=False)
    new_values, new_action_log_probs, dist_entropy = trainer.evaluate_actions(obs, actions)
    assert env.envs[0].action_space.shape[0] == actions.shape[1]
    assert values.shape == (3, 1), values.shape
    assert action_log_probs.shape == (3, 1), action_log_probs.shape
    assert dist_entropy.shape == ()
    assert (values == new_values).all()
    assert (action_log_probs == new_action_log_probs).all()
    assert dist_entropy.shape == ()

    print("Base trainer continuous case test passed!")
    env.close()


if __name__ == '__main__':
    test_base_trainer()
    print("Base trainer test passed!")
