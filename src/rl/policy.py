"""This code is built based on ActorCriticPolicy implementation from stable_baseline3"""

from typing import Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
from stable_baselines3.common.policies import (ActorCriticPolicy,
                                               MultiInputActorCriticPolicy)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import (get_device,
                                            is_vectorized_observation,
                                            obs_as_tensor)
from utils import MLP
import torch.nn.functional as F


class MlpExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = torch.device(device)

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = MLP(net_arch, emb_size=feature_dim)
        # self.value_net = MLP(net_arch, output_size=1, emb_size=feature_dim) ## TODO: figure this out # nn.Sequential(*value_net).to(device)
        # self.value_net = nn.Sequential(*[
        #     nn.Linear(1025, 64),
        #     nn.Linear(64, 128), nn.GELU(),
        #     nn.Linear(128, 128), nn.GELU(),
        #     # nn.Linear(128, 64), nn.GELU(),
        #     # nn.Linear(64, 32), nn.GELU(),
        #     # nn.Linear(32, 16), nn.GELU(),
        #     # nn.Linear(16, 1), nn.GELU(),
        # ])
        self.value_net = MLP(net_arch, emb_size=feature_dim)
        self.latent_dim_pi = net_arch[-1]
        self.latent_dim_vf = net_arch[-1]

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        output = self.policy_net(features)
        return output

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class MLPPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MLPPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device
        )

    def evaluate_actions(self, obs: torch.Tensor | Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor | None]:
        return super().evaluate_actions(obs, actions)

    # def _build(self, lr_schedule: Schedule) -> None:
    #     """
    #     Create the networks and the optimizer.

    #     :param lr_schedule: Learning rate schedule
    #         lr_schedule(1) is the initial learning rate
    #     """

    #     self.value_net =



class Policy(MultiInputActorCriticPolicy):


    @torch.no_grad()
    def evaluate_actions(self, obs, actions, marginal_network):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()

        observation, t = obs["obs"], obs["time"]
        # NOTE: 
        # 1. time embedding shape
        # 2. potentially memory issues since all buffers are on GPU
        output = marginal_network(
            observation, 
            t.reshape(-1, 1)
        )
        rewards = F.mse_loss(output, actions)
        
        return values, log_prob, entropy, rewards
