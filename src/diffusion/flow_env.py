from typing import *
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.utils as utils
from gymnasium import spaces
from gymnasium.utils import seeding
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import v2
from utils import SmileyFaceDataset, MLP

from flow_matching import EmpiricalFlowMatching


class FlowDiffusionEnv(gym.Env):
    def __init__(
        self,
        dataset: str,
        obs_horizon: Tuple[float],
        obs_shape: Tuple[float],
        action_range: Tuple[float],
        action_shape: Tuple[float],
        max_time_step: int,
        batch_size=256
    ) -> None:
        super(FlowDiffusionEnv, self).__init__()
        transform = v2.Compose([v2.ToTensor()])
        if dataset not in {"MNIST", "CIFAR10", "smiley_face"}:
            raise Exception("OOF, INVALID DATASET")
        elif dataset == "smiley_face":
            data_file = Path("./data/smiley_dataset.pkl")
            if data_file.exists():
                self.dataset = SmileyFaceDataset(data_path=str(data_file.resolve()))
            else:
                self.dataset = SmileyFaceDataset(transform=transform)
                self.dataset.save(str(data_file.resolve()))
        else:
            self.dataset = eval(
                f"torchvision.datasets.{dataset}(root='./data/', transform=transform)")
        self.observation_space = spaces.Box(*obs_horizon, obs_shape)
        self.action_space = spaces.Box(*action_range, action_shape)
        self.batch_size = batch_size
        self.time = 0
        self.max_time_step = max_time_step

        # current images
        self.cur_state = None
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
    def step(self, action: torch.Tensor, value_net: nn.Module):
        """
        Args:
            action [torch.Tensor]: same dimension with image space, predict the movement of the image
            value_net [nn.Module]: value network for estimating MSE Loss with forward learned vector field network
        """
        is_terminated = False
        is_truncated = False
        info = dict()

        is_terminated = self._is_terminated()
        is_truncated = self._is_truncated()
        
        # TODO: fix updating current state and maintaining states 
        self.cur_state = self.cur_state + action
        self.states.append(self.cur_state)
        self.time += 1
        reward = self._calculate_reward(action)

        return self.cur_state, reward, is_terminated, is_truncated, info

    def reset(self, seed=None, options=None):
        self.img_idxs = torch.randperm(self.dataset.shape[0])[:self.batch_size]
        self.states = self.dataset[self.img_idxs]
        self.time = 0
        return self.states, dict()

    def render(self, *args, **kwargs):
        pass

    def close(self):
        # Don't really need to worry about this
        pass

    @torch.no_grad()
    def _calculate_reward(self, action: torch.Tensor):
        true_direction = self.states[self.time + 1] - self.states[self.time]
        return torch.norm(true_direction - action)

    def _is_terminated(self):
        return torch.norm(self.cur_state - self.dataset[self.img_idxs]) < 0.01

    def _is_truncated(self):
        return self.time >= self.max_time_step
