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


class ResNetBlock(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.fc = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.fc(x))


class MLP(nn.Module):
    def __init__(self, net_arch=[128, 128, 128], emb_size: int = 128):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size)
        self.input_mlp1 = PositionalEmbedding(emb_size, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, scale=25.0)

        # size of embeddings with input data
        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        
        # First layer of neurons
        layers = [nn.Linear(concat_size, net_arch[0]), nn.GELU()]

        # Hidden Layers
        for i in range(1, len(net_arch)):
            layers.append(ResNetBlock(net_arch[i]))

        # Output layer, project back to original data dimension
        layers.append(nn.Linear(net_arch[-1], 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)

        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x
