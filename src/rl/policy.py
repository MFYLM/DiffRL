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

    # def _build(self, lr_schedule: Schedule) -> None:
    #     """
    #     Create the networks and the optimizer.

    #     :param lr_schedule: Learning rate schedule
    #         lr_schedule(1) is the initial learning rate
    #     """

    #     self.value_net =
