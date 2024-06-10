from pathlib import Path
from typing import *

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils as utils
from gymnasium import spaces
from gymnasium.utils import seeding
from matplotlib import pyplot as plt
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import v2
from utils import MLP, SmileyFaceDataset, SpiralDataset
from typing import Tuple, List

from .flow_matching import EmpiricalFlowMatching


class FlowDiffusionEnv(gym.Env):
    def __init__(
        self,
        marginal_network: nn.Module,
        marginal_optimizer: optim.Optimizer,
        dataset: str,
        obs_range: Tuple[float],
        # obs_horizon: int,
        obs_shape: Tuple[float],
        action_range: Tuple[float],
        action_shape: Tuple[float],
        max_time_steps: int,
    ) -> None:
        super(FlowDiffusionEnv, self).__init__()
        self.marginal_network = marginal_network
        self.marginal_optimizer = marginal_optimizer

        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        if dataset not in {"MNIST", "CIFAR10", "smiley_face", "spiral"}:
            raise Exception("OOF, INVALID DATASET")
        elif dataset == "smiley_face":
            data_file = Path("./data/smiley_dataset.pkl")
            if data_file.exists():
                self.dataset = SmileyFaceDataset(data_path=str(data_file.resolve()))
            else:
                self.dataset = SmileyFaceDataset(transform=transform, data_path=None)
                self.dataset.save(str(data_file.resolve()))
        elif dataset == "spiral":
            data_file = Path("./data/spiral_dataset.pkl")
            if data_file.exists():
                self.dataset = SpiralDataset(data_path=str(data_file.resolve()))
            else:
                self.dataset = SpiralDataset(transform=transform, data_path=None)
                self.dataset.save(str(data_file.resolve()))
        else:
            self.dataset = eval(
                f"{dataset}(root='./data/', transform=transform, download=True)"
            )
        self.observation_space = gym.spaces.Dict(
            {
                "initial": spaces.Box(*obs_range, obs_shape),
                "current": spaces.Box(*obs_range, obs_shape),
                "final": spaces.Box(*obs_range, obs_shape),
                "time": spaces.Box(0, max_time_steps, (1,), int),
            }
        )
        self.action_space = spaces.Box(*action_range, action_shape)
        self.max_time_steps = max_time_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.maybe_plot_final_marginal = False
        self.dt = (1.0 - 0.0) / self.max_time_steps
        self.time = torch.zeros((1,))

        # current images
        self.cur_state = torch.randn(*action_shape)
        # all images in the sampled trajectory
        self.states = []
        # action log probabilities for the sampled trajectory
        self.action_log_probs = []

        # rewards for the sampled trajectory
        self.rewards = []
        # termination for the sampled trajectory
        self.is_terminated = []
        self.img_idxs = []

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        """
        Args:
            action [torch.Tensor]: same dimension with image space, predict the movement of the image
            value_net [nn.Module]: value network for estimating MSE Loss with forward learned vector field network
        """
        is_terminated = self._is_terminated()
        is_truncated = self._is_truncated()

        self.cur_state = self.cur_state + self.dt * action
        # self.states.append(self.cur_state)
        reward = self._calculate_reward(action)
        self.time += self.dt
        return (
            {
                "initial": self.init_states,
                "current": self.cur_state,
                "final": self.final_state,
                "time": self.time,
            },
            reward,
            is_terminated,
            is_truncated,
            {},
        )

    def reset(self, seed=None, options=None):
        self.img_idxs = torch.randint(0, len(self.dataset), (1,))
        self.init_states, self.final_state = self.dataset[self.img_idxs]
        self.cur_state = self.init_states.to(self.device)
        self.final_state = self.final_state.to(self.device)
        self.time = torch.zeros((1,), device=self.device)
        return {
            "initial": self.init_states,
            "current": self.cur_state,
            "final": self.final_state,
            "time": self.time,
        }, {}

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def update_marginal(self, conditional_model: nn.Module):
        # conditional_model_input = torch.concatenate((
        #     self.init_states.flatten(),
        #     self.final_state.flatten(),
        #     self.time.flatten())).reshape(1,-1)
        obs = {
            "initial": self.init_states.unsqueeze(0),
            "current": self.cur_state.unsqueeze(0),
            "final": self.final_state.unsqueeze(0),
            "time": self.time.unsqueeze(0),
        }
        with torch.no_grad():
            conditional_action, values, log_prob = conditional_model(obs)
        print("current state flat:", self.cur_state.flatten().unsqueeze(0).shape)
        marginal_action = self.marginal_network(self.cur_state.flatten().unsqueeze(0), self.time)
        print("conditional action:", conditional_action.shape)
        
        # FIXME: marginal action output size: (128,) -> expected (512, 2)
        print("marginal action:", marginal_action.shape)
        loss = (log_prob * (conditional_action - marginal_action) ** 2).mean()
        loss.backward()
        self.marginal_optimizer.step()

    @torch.no_grad()
    def _calculate_reward(self, conditional_action: torch.Tensor):
        marginal_action = self.marginal_network(self.cur_state, self.time)
        return F.mse_loss(marginal_action, conditional_action)

    def _is_terminated(self):
        return F.mse_loss(self.cur_state - self.final_state) < 1e-4

    def _is_truncated(self):
        return self.time >= self.max_time_steps
