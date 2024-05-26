import torch

from rl import PPO, MLPPolicy
from diffusion import FlowDiffusionEnv
from utils import parse_args

import time, matplotlib

import matplotlib.pyplot as plt

def train(args):
    env = FlowDiffusionEnv(
        **args.env_kwargs
    )

    model = PPO(
        policy=MLPPolicy,
        env=env,
        policy_kwargs={"net_arch":[64, 128]},
    )

    model.learn(total_timesteps=100)



def env_test(args):
    env = FlowDiffusionEnv(
        **args.env_kwargs
    )

    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()

        plt.figure()
        plt.imshow(obs[0].squeeze())
        plt.show(block=True)
        time.sleep(3)

        obs, reward, done, trunc, info = env.step(action)

        if done or trunc:
            break




def main():
    args = parse_args()

    metrics = train(args)


if __name__ == "__main__":
    main()