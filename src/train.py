import torch

from rl import PPO
from diffusion import FlowDiffusionEnv
from utils import parse_args

def train(args):

    env = FlowDiffusionEnv(
        **args.env_kwargs
    )

    PPO = PPO(
        policy_net="MlpPolicy",
        env=env,
    )

def main():
    args = parse_args()

    metrics = train(args)


if __name__ == "__main__":
    main()