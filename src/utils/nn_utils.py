from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from zuko.utils import odeint
from typing import Union, Dict


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0]).to(x.device)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size).to(x.device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, **kwargs):
        super().__init__()

        self.layer = SinusoidalEmbedding(size, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class LearnedEmbedding(nn.Module):
    def __init__(self, input_dim: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, emb_size)

    def forward(self, x: torch.Tensor):
        return self.embedding(x)


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.fc = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.fc(x))


class MLP(nn.Module):
    # output_size was default 2
    def __init__(
        self, input_size: int, net_arch=[128, 128, 128], output_size: int = 128, emb_size: int = 128
    ):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size)

        # size of embeddings with input data
        concat_size = len(self.time_mlp.layer) + input_size # + \
        # len(self.input_mlp1.layer) + len(self.input_mlp2.layer)

        # First layer of neurons
        layers = [nn.Linear(concat_size, net_arch[0]), nn.GELU()]

        # Hidden Layers
        for i in range(1, len(net_arch)):
            layers.append(Block(net_arch[i]))

        # Output layer, project back to original data dimension
        layers.append(nn.Linear(net_arch[-1], output_size))
        self.joint_mlp = nn.Sequential(*layers)

    @torch.autocast(device_type="cpu", dtype=torch.float32)
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = torch.cat((x, t_emb), dim=-1)
        if x.size(0) == 1:
            x = x.flatten()
        print("joint mlp: ", list(list(self.joint_mlp.children())[0].parameters())[0].dtype)
        print(f"x dtype: {x.dtype}")
        # TODO: why does this network expect double
        x = self.joint_mlp(x)
        return x


class ReinforceMLP(nn.Module):
    # output_size was default 2
    def __init__(
        self, net_arch=[128, 128, 128], output_size: int = 128, emb_size: int = 128
    ):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size - 1)

        # size of embeddings with input data
        concat_size = len(self.time_mlp.layer) * 2  # + \
        # len(self.input_mlp1.layer) + len(self.input_mlp2.layer)

        # First layer of neurons
        layers = [nn.Linear(concat_size, net_arch[0]), nn.GELU()]

        # Hidden Layers
        for i in range(1, len(net_arch)):
            layers.append(Block(net_arch[i]))

        # Output layer, project back to original data dimension
        layers.append(nn.Linear(net_arch[-1], output_size))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = torch.cat((x, t_emb), dim=-1)
        if x.size(0) == 1:
            x = x.flatten()
        x = self.joint_mlp(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, fc: nn.Module, act: nn.Module = nn.GELU()):
        super().__init__()

        self.fc = fc
        self.act = act

    def forward(self, x: torch.Tensor):
        return x + self.act(self.fc(x))


"RL Agent"


class ConditionalVectorField(nn.Module):
    def __init__(
        self, 
        input_size: int,
        step: float,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        device: torch.device=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.step = step
        self.drift_net = MLP(input_size, net_arch, emb_size=feature_dim).to(self.device)
        self.covariance_net = MLP(input_size, emb_size=feature_dim).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, t = x[:,:-1], x[:,-1]
        drift = self.drift_net(x, t)
        covariance = self.covariance_net(x, t)
        print("self.step:", type(self.step))
        return drift * self.step + covariance * (self.step ** 0.5)


class VectorField(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, h_dims: List[int], embed_dim: int
    ) -> None:
        super().__init__()

        self.time_mlp = PositionalEmbedding(embed_dim - 1)

        concat_size = len(self.time_mlp.layer) + in_dim
        ins = [concat_size] + h_dims
        outs = h_dims + [out_dim]

        self.layers = nn.ModuleList(
            [
                nn.Sequential(ResnetBlock(nn.Linear(in_d, out_d), nn.GELU()))
                for in_d, out_d in zip(ins, outs)
            ]
        )
        self.top = nn.Sequential(nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, t = x[:, :-1], x[:, -1]
        t = self.time_mlp(t)
        x = torch.cat((x, t), dim=-1)

        for l in self.layers:
            x = l(x)
        return self.top(x)


class ReinforcePolicy(nn.Module):
    def __init__(
        self,
        input_dim: torch.Tensor,
        output_dim: torch.Tensor,
        activation: str = "gelu",
    ) -> None:
        super(ReinforcePolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc_mean = nn.Linear(32, output_dim)
        self.fc_std = nn.Linear(32, output_dim)
        self.activation = eval(f"F.{activation}")

    def forward(self, x: torch.Tensor):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        return mean, std

    def predict(self, obs: torch.Tensor):
        mean, std = self.forward(obs)
        return self.select_action(mean, std)

    def select_action(self, mean, std):
        normal_dist = torch.distributions.Normal(mean, std)
        action = normal_dist.sample((512, 2))
        lp = normal_dist.log_prob(action)
        return action, lp


