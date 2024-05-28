import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusion import FlowDiffusionEnv
from rl import PPO, MLPPolicy
from utils import parse_args


def train(args):
    env = FlowDiffusionEnv(
        **args.env_kwargs
    )

    model = PPO(
        policy=MLPPolicy,
        env=env,
        policy_kwargs={"net_arch": [128, 128, 128]},
        gamma=1
    )

    model.learn(total_timesteps=10000)

    imgs = []
    rewards = []

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs)

        obs, reward, done, trunc, info = env.step(action)
        rewards.append(reward)
        # print("done, trunc:", done.item(), trunc.item())

        imgs.append(obs["obs"].reshape(512, 2))

        if done or trunc:
            break

    plt.plot(np.arange(len(rewards)), rewards)
    plt.show(block=True)
    for img in imgs[-10:]:
        plt.scatter(img[:, 0], img[:, 1])
        plt.show(block=True)


def main():
    args = parse_args()

    metrics = train(args)


if __name__ == "__main__":
    main()
