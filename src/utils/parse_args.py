import argparse as ap

def construct_env_kwargs(args):
    env_kwargs = {
        "max_time_step": args.max_time_steps,
        # "batch_size": args.batch_size,
        "action_range": (-0.1, 0.1),
        # "obs_horizon": args.obs_horizon,
    }
    if args.dataset == "MNIST":
        env_kwargs.update({
            "dataset": "MNIST",
            "obs_shape": (1, 28, 28), # (args.obs_horizon, args.batch_size, 1, 28, 28),
            "action_shape": (1, 28, 28), # (args.batch_size, 1, 28, 28),
            "obs_range": (0, 1),
        })
    elif args.dataset == "CIFAR10":
        env_kwargs.update({
            "dataset": "CIFAR10",
            "obs_shape": (3, 32, 32), # (args.obs_horizon, args.batch_size, 3, 32, 32),
            "action_shape": (3, 32, 32), # (args.batch_size, 3, 32, 32),
            "obs_range": (0, 1),
        })
    elif args.dataset == "smiley_face":
        # only for points (512, 2)
        env_kwargs.update({
            "dataset": "smiley_face",
            "obs_shape": (512, 2), # (args.obs_horizon, args.batch_size, 512, 2),
            "action_shape": (512, 2), # (args.batch_size, 2),
            "obs_range": (0, 5),
        })
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    return env_kwargs
    

def parse_args():

    parser = ap.ArgumentParser(description="Train a FlowDiffusion model")

    parser.add_argument("-t", "--max_time_steps", type=int, default=1000)
    # parser.add_argument("-bs", "--batch_size", type=int, default=32)
    # parser.add_argument("-oh", "--obs_horizon", type=int, default=1)
    parser.add_argument(
        "-d", "--dataset", type=str, choices=["MNIST", "smiley_face", "CIFAR10"], default="smiley_face"
    )

    args = parser.parse_args()
    args.env_kwargs = construct_env_kwargs(args)

    return args