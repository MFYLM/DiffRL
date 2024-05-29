import torch
from diffusion import FlowDiffusionEnv
from utils import MLP, ReinforceMLP, parse_args

def reinforce():
    pass

def train(args):
    env = FlowDiffusionEnv(
        **args.env_kwargs
    )
    policy = MLP(net_arch=[128,128,128])

    obs_data, _ = env.reset()
    while True:
        obs, time = obs_data["obs"], obs_data["time"]
        print(obs.flatten().unsqueeze(0).shape, time.shape)
        action = policy(obs.flatten().unsqueeze(0), time)
        print("action:", action.shape)
        break

if __name__ == "__main__":
    args = parse_args()
    metrics = train(args)