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
        policy_kwargs={"net_arch":[128, 128, 128]},
    )

    model.learn(total_timesteps=10000)

    imgs = []

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs)

        obs, reward, done, trunc, info = env.step(action)

        imgs.append(obs["obs"].reshape(32, 32))

        if done or trunc:
            break

    import numpy as np
    table = np.arange(1000).reshape(100, 10) + 1
    print(imgs[0].shape)
    for img in imgs[-10:]:
        plt.imshow(img)
        plt.show(block=True)

def main():
    args = parse_args()

    metrics = train(args)


if __name__ == "__main__":
    main()