from pathlib import Path
from typing import *

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.utils as utils
from gymnasium import spaces
from gymnasium.utils import seeding
from matplotlib import pyplot as plt
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import v2
from utils import MLP, SmileyFaceDataset, create_flow_matching

from .flow_matching import EmpiricalFlowMatching


class FlowDiffusionEnv(gym.Env):
    def __init__(
        self,
        dataset: str,
        obs_range: Tuple[float],
        # obs_horizon: int,
        obs_shape: Tuple[float],
        action_range: Tuple[float],
        action_shape: Tuple[float],
        max_time_step: int,
        weights_path: str,
        # batch_size=256
    ) -> None:
        super(FlowDiffusionEnv, self).__init__()
        transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        if dataset not in {"MNIST", "CIFAR10", "smiley_face"}:
            raise Exception("OOF, INVALID DATASET")
        elif dataset == "smiley_face":
            data_file = Path("./data/smiley_dataset.pkl")
            if data_file.exists():
                self.dataset = SmileyFaceDataset(
                    data_path=str(data_file.resolve()))
            else:
                self.dataset = SmileyFaceDataset(
                    transform=transform, data_path=None)
                self.dataset.save(str(data_file.resolve()))
        else:
            self.dataset = eval(
                f"{dataset}(root='./data/', transform=transform, download=True)")
        self.observation_space = gym.spaces.Dict({
            "obs": spaces.Box(*obs_range, obs_shape),
            "time": spaces.Box(0, max_time_step, (1, ), int)
        })
        self.action_space = spaces.Box(*action_range, action_shape)
        # self.batch_size = batch_size
        self.time = 0
        self.max_time_step = max_time_step

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.marginal_model = create_flow_matching(self.device)
        with open(weights_path, 'rb') as f:
            self.marginal_model.load_state_dict(torch.load(f, map_location=self.device))
        self.marginal_states = []
        t0 = torch.zeros((512, 2))[:,0].to(self.device)
        t1 = torch.ones((512, 2))[:,0].to(self.device)
        self.h = (t1 - t0)/self.max_time_step
        self.t = t0

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

        loc = torch.zeros(*action_shape).flatten()
        self.init_dist = torch.distributions.multivariate_normal.MultivariateNormal(
            loc, torch.eye(loc.shape[0], loc.shape[0]))

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
        is_terminated = False
        is_truncated = False

        is_terminated = self._is_terminated()
        is_truncated = self._is_truncated()

        self.cur_state = self.cur_state + action
        self.states.append(self.cur_state)
        # self.states.append(torch.tensor(action))
        reward = self._calculate_reward(action)
        self.time += 1

        return {"obs": self.cur_state, "time": self.time}, reward, is_terminated, is_truncated, {}

    def reset(self, seed=None, options=None):
        # self.img_idxs = torch.randperm(len(self.dataset))[:self.batch_size]
        self.img_idxs = torch.randint(0, len(self.dataset), (1, ))
        # subset = Subset(self.dataset, self.img_idxs)
        # self.orig = self.cur_state = torch.stack([img for img, _ in subset])
        self.orig, _ = self.dataset[self.img_idxs]
        self.cur_state = self.init_dist.rsample().reshape(*self.orig.shape)
        self.states = [self.cur_state]
        self.marginal_states = [torch.randn((512, 2)).to(self.device)]
        self.time = torch.zeros(1, dtype=int)
        return {"obs": self.cur_state, "time": self.time}, {}

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass

    @torch.no_grad()
    def _calculate_reward(self, action: torch.Tensor):
        true_direction = self.h[:,None] * self.marginal_model(self.t, self.marginal_states[-1]).squeeze()
        updated_marginal_state = self.marginal_states[-1] + true_direction
        self.t = self.t + self.h
        # copy = updated_marginal_state.clone().squeeze().detach().cpu().numpy()
        # plt.scatter(copy[:,0], copy[:,1])
        # plt.savefig(f"./all_imgs/{self.time.item()}.jpg")
        # plt.close()
        # if self.time.item() == 1000:
        #     print("done saving")
        #     input()
        self.marginal_states.append(updated_marginal_state)
        dist = -torch.norm(true_direction.flatten() - action.flatten())
        # dist = -torch.norm(self.marginal_states[-1].flatten() - self.states[-1].flatten())
        return dist

    def _is_terminated(self):
        return torch.norm(self.cur_state - self.orig) < 0.01

    def _is_truncated(self):
        return self.time >= self.max_time_step
