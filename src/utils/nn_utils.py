from typing import List

import torch
import torch.nn as nn


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
