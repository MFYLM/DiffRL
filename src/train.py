from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from diffusion import FlowDiffusionEnv
from rl import PPO, MLPPolicy, EmpiricalFlowMatchingPolicy
from utils import parse_args, MLP
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    marginal_network = MLP(input_size=1024).to(device)
    marginal_optimizer = optim.Adam(marginal_network.parameters())
    env = FlowDiffusionEnv(
        marginal_network,
        marginal_optimizer,
        **args.env_kwargs
    )

    model = PPO(
        policy=EmpiricalFlowMatchingPolicy,
        env=env,
        marginal_network=marginal_network,
        marginal_optimizer=marginal_optimizer,
        policy_kwargs={"input_size": 2048, "net_arch": [128, 128, 128], "step": 1.0 / args.env_kwargs["max_time_steps"], "feature_dim": 128},
        gamma=1
    )

    # 50 timesteps per episode, a 1000000 total timesteps
    model.learn(total_timesteps=10000)

    # imgs = []
    # marginals = []
    rewards = []
    # actions = []

    obs, _ = env.reset()
    all_obs = [obs["obs"]]
    while True:
        action, _ = model.predict(obs)
        # actions.append(action.reshape(512, 2))
        obs, reward, done, trunc, info = env.step(action)
        # all_obs.append(obs["obs"])
        # rewards.append(reward)
        # imgs.append(obs["obs"].reshape(512, 2))
        # marginals.append(env.marginal_states[-1].reshape(512,2).detach().cpu().numpy())
        if done or trunc:
            break

    
    # runs_folder = Path("../../runs")
    # runs_folder.mkdir(exist_ok=True)

    # n_runs = len([i for i in runs_folder.iterdir()])
    # run_folder = Path(f"../../runs/run_{n_runs}")
    # run_folder.mkdir(exist_ok=False, parents=True)
    # run_folder_path = str(run_folder.resolve())

    # plt.plot(np.arange(len(rewards)), rewards)
    # plt.savefig(f"{run_folder_path}/rewards.jpg")
    # plt.close()

    # for i, img in enumerate(imgs):
    #     fig, axes = plt.subplots(2, 2, figsize=(8,8))
        
    #     # Scatter plots for marginals and imgs
    #     axes[0, 0].set_title("Marginal Vector Field's Points")
    #     axes[0, 0].scatter(marginals[i][:, 0], marginals[i][:, 1])
    #     axes[0, 1].set_title("Policy's Points")
    #     axes[0, 1].scatter(img[:, 0], img[:, 1])
        
    #     # Prepare data for quiver plot
    #     obs = all_obs[i].squeeze()
    #     t_obs = env.marginal_states[i]
    #     actions_i = actions[i].squeeze()
    #     true_actions = env.true_directions[i].squeeze()
    #     obs_x, obs_y, dx, dy = obs[:, 0], obs[:, 1], actions_i[:,0], actions_i[:,1]
    #     t_obs_x, t_obs_y, t_dx, t_dy = t_obs[:, 0], t_obs[:, 1], true_actions[:,0], true_actions[:,1]
        
    #     # Debug print to check shapes
    #     # print(obs.shape, obs_x.shape, obs_y.shape, dx.shape, dy.shape)
        
    #     # Ensure shapes are compatible for quiver
    #     if obs_x.shape == dx.shape and obs_y.shape == dy.shape:
    #         axes[1, 0].set_title("Marginal Vector Field's Vector Field")
    #         axes[1,0].quiver(t_obs_x, t_obs_y, t_dx, t_dy)
    #         axes[1, 1].set_title("Policy's Vector Field")
    #         axes[1,1].quiver(obs_x, obs_y, dx, dy)
    #     else:
    #         print(f"Shapes are not compatible for quiver plot at index {i}")
        
    #     # Save figure
    #     plt.savefig(f"{run_folder_path}/img{i}.jpg")
    #     plt.close()

    # # Plot final image and save
    # fig, axes = plt.subplots(2, 2, figsize=(8,8))
    # axes[0, 0].set_title("Marginal Vector Field's Points")
    # axes[0, 0].scatter(marginals[-1][:, 0], marginals[-1][:, 1])
    # axes[0, 1].set_title("Policy's Points")
    # axes[0, 1].scatter(imgs[-1][:, 0], imgs[-1][:, 1])
    # obs = all_obs[-1].squeeze()
    # t_obs = env.marginal_states[-1]
    # actions_i = actions[-1].squeeze()
    # true_actions = env.true_directions[-1].squeeze()
    # obs_x, obs_y, dx, dy = obs[:, 0], obs[:, 1], actions_i[:,0], actions_i[:,1]
    # t_obs_x, t_obs_y, t_dx, t_dy = t_obs[:, 0], t_obs[:, 1], true_actions[:,0], true_actions[:,1]
    # axes[1, 0].set_title("Marginal Vector Field's Vector Field")
    # axes[1,0].quiver(t_obs_x, t_obs_y, t_dx, t_dy)
    # axes[1, 1].set_title("Policy's Vector Field")
    # axes[1,1].quiver(obs_x, obs_y, dx, dy)
    # plt.savefig(f"{run_folder_path}/img_final.jpg")
    # plt.close()


def main():
    args = parse_args()

    metrics = train(args)


if __name__ == "__main__":
    main()
