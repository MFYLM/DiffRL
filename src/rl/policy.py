from utils import MLP

from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn

from typing import Union, List, Dict, Type, Tuple

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
        self.value_net = None ## TODO: figure this out # nn.Sequential(*value_net).to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class MLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MLPPolicy, self).__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        return MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device
        )

