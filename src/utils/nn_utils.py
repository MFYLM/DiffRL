from typing import List

import torch
import torch.nn as nn

from zuko.utils import odeint

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
    def __init__(self, size: int,  **kwargs):
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
    def __init__(self, net_arch=[128, 128, 128], output_size: int = 128, emb_size: int = 128):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size - 1)
        # self.input_mlp1 = PositionalEmbedding(emb_size - 1, scale=25.0)
        # self.input_mlp2 = PositionalEmbedding(emb_size - 1, scale=25.0)

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

    def forward(self, x):
        x, t = x[:, :-1], x[:, -1]
        # x1_emb = self.input_mlp1(x[:, 0])
        # x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        # x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
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
    def __init__(self, in_dim: int, out_dim: int, h_dims: List[int], embed_dim: int) -> None:
        super().__init__()

        self.time_mlp = PositionalEmbedding(embed_dim - 1)
        self.in_dim = in_dim
        concat_size = len(self.time_mlp.layer) + in_dim * 2
        ins = [concat_size] + h_dims
        outs = h_dims + [out_dim]

        self.layers = nn.ModuleList([
            nn.Sequential(ResnetBlock(nn.Linear(in_d, out_d), nn.GELU())) for in_d, out_d in zip(ins, outs)
        ])
        self.top = nn.Sequential(nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, condition, t = x[:, :self.in_dim], x[:, self.in_dim:-1], x[:, -1]
        t = self.time_mlp(t)
        x = torch.cat((x, condition, t), dim=-1)

        for l in self.layers:
            x = l(x)
        return self.top(x)


"The Marginal that is fixed during RL training"
#@title ⏳ Summary: please run this cell which contains the ```OTFlowMatching``` class
class OTFlowMatching:

  def __init__(self, sig_min: float = 0.001) -> None:
    super().__init__()
    self.sig_min = sig_min
    self.eps = 1e-5

  def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """ Conditional Flow
    """
    return (1 - (1 - self.sig_min) * t) * x + t * x_1


class VectorField(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, h_dims: List[int], embed_dim: int) -> None:
        super().__init__()

        self.time_mlp = PositionalEmbedding(embed_dim - 1)

        concat_size = len(self.time_mlp.layer) + in_dim
        ins = [concat_size] + h_dims
        outs = h_dims + [out_dim]

        self.layers = nn.ModuleList([
            nn.Sequential(ResnetBlock(nn.Linear(in_d, out_d), nn.GELU())) for in_d, out_d in zip(ins, outs)
        ])
        self.top = nn.Sequential(nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, t = x[:, :-1], x[:, -1]
        t = self.time_mlp(t)
        x = torch.cat((x, t), dim=-1)

        for l in self.layers:
            x = l(x)
        return self.top(x)

class CondVF(nn.Module):
  def __init__(self, net: nn.Module, n_steps: int = 100) -> None:
    super().__init__()
    self.net = net

  def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return self.net(t, x)

  def wrapper(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
      t = t * torch.ones(len(x), device=x.device)
      return self(t, x)

  def decode_t0_t1(self, x_0, t0, t1):
    return odeint(self.wrapper, x_0, t0, t1, self.parameters())

  def encode(self, x_1: torch.Tensor) -> torch.Tensor:
    return odeint(self.wrapper, x_1, 1., 0., self.parameters())

  def decode(self, x_0: torch.Tensor) -> torch.Tensor:
    return odeint(self.wrapper, x_0, 0., 1., self.parameters())

def create_flow_matching(device: torch.device):
    net = VectorField(2, 2, [512]*5, 10).to(device)
    v_t = CondVF(net)
    return v_t

def calculate_loss(self, v_t: nn.Module, x_1: torch.Tensor) -> torch.Tensor:
    """ Compute loss
    """
    # t ~ Unif([0, 1])
    t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
    t = t[:, None].expand(x_1.shape)
    # x ~ p_t(x_0)
    x_0 = torch.randn_like(x_1)
    v_psi = v_t(t[:,0], self.psi_t(x_0, x_1, t))
    d_psi = x_1 - (1 - self.sig_min) * x_0
    return torch.mean((v_psi - d_psi) ** 2)