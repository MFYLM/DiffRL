import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

from sklearn.datasets import make_swiss_roll

class SmileyFaceDataset(torch.utils.data.Dataset):
    def __init__(self, size: int = 100, transform=None, data_path: str = None):
        if data_path is not None:
            self.load(data_path)
        else:
            self.data = []
            img = generate_happy_face(num_samples=512)
            if transform:
                img = transform(img)
            for _ in range(size):
                self.data.append(img)

    def load(self, data_path):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def save(self, data_path):
        with open(data_path, "wb") as f:
            pickle.dump(self.data, f)

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return len(self.data)


def ode_solve(z0, t0, t1, f):
    n_steps = 1000

    h = (t1 - t0) / n_steps
    t = t0
    z = z0

    zs = []

    for i_step in range(n_steps):
        z = z + h[:, None] * f(z, t)
        zs += [z.cpu().detach().numpy()]
        t = t + h

    return zs

class SpiralDataset(torch.utils.data.Dataset):
    def __init__(self, size: int = 100, transform=None, data_path: str = None):
        if data_path is not None:
            self.load(data_path)
        else:
            self.data = []
            data, _ = make_swiss_roll(n_samples=512, noise=0.25)
            img = data[:,[0,2]]/10.0
            if transform:
                img = transform(img)
            for _ in range(size):
                self.data.append(img)

    def load(self, data_path):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def save(self, data_path):
        with open(data_path, "wb") as f:
            pickle.dump(self.data, f)

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return len(self.data)


def generate_mixture_gaussians(num_samples=32, centers=6, spread=.5, radius=5.0):
    """
    Generates a mixture of 2D Gaussian distributions around the origin (0,0).

    :param num_samples: Total number of samples to generate.
    :param centers: Number of Gaussian centers.
    :param spread: Standard deviation of each Gaussian.
    :param radius: Radius to space out the Gaussian centers on a circle.
    :return: Tensor of shape (num_samples, 2) containing the Gaussian mixture data.
    """
    data = []
    samples_per_center = num_samples // centers
    remaining_samples = num_samples - samples_per_center * centers

    angles = np.linspace(0, 2 * np.pi, centers, endpoint=False)

    for i, angle in enumerate(angles):

        if i == len(angles) - 1:
            samples_per_center += remaining_samples

        # Calculate center of Gaussian
        center_x = radius * np.cos(angle)
        center_y = radius * np.sin(angle)
        center = np.array([center_x, center_y])

        # Generate samples for a Gaussian
        samples = np.random.normal(
            loc=center, scale=spread, size=(samples_per_center, 2))
        data.append(samples)

    # Concatenate all samples and shuffle
    data = np.vstack(data)
    np.random.shuffle(data)

    return torch.tensor(data, dtype=torch.float32)


def generate_happy_face(num_samples=32, spread=0.01):
    """
    Generates a dataset shaped like a happy face with clearer features.
    :param num_samples: Total number of samples to generate.
    :param spread: Standard deviation for the Gaussian noise added to the features.
    :return: Tensor of shape (num_samples, 2) containing the happy face data.
    """
    # Create an empty list to hold data points
    data_points = []

    # Face outline - circle
    face_samples = num_samples // 2
    angles = np.linspace(0, 2 * np.pi, face_samples)
    x_face = np.cos(angles)
    y_face = np.sin(angles)
    data_points.extend(zip(x_face, y_face))

    # Eyes - two smaller circles
    eye_samples = num_samples // 20
    eye_radius = 0.1
    angles = np.linspace(0, 2 * np.pi, eye_samples)

    # Left eye
    x_left_eye = -0.35 + eye_radius * np.cos(angles)
    y_left_eye = 0.45 + eye_radius * np.sin(angles)
    data_points.extend(zip(x_left_eye, y_left_eye))

    # Right eye
    x_right_eye = 0.35 + eye_radius * np.cos(angles)
    y_right_eye = 0.45 + eye_radius * np.sin(angles)
    data_points.extend(zip(x_right_eye, y_right_eye))

    # Smile - semi-circle
    smile_samples = num_samples - (face_samples + 2 * eye_samples)
    angles = np.linspace(0.75 * np.pi, 0.25 * np.pi, smile_samples)
    x_smile = -0.85 * np.cos(angles)
    y_smile = -0.85 * np.sin(angles) + 0.35
    data_points.extend(zip(x_smile, y_smile))

    # Convert data_points to numpy array and shuffle
    data = np.array(data_points)
    np.random.shuffle(data)

    # Add Gaussian noise
    data += np.random.normal(scale=spread, size=data.shape)

    return torch.tensor(data*6, dtype=torch.float32)


def generate_path_data(original_imgs: torch.Tensor, target_imgs: torch.Tensor) -> torch.Tensor:
    """
    Generate sample path for given set of images and record necessary information
    along the way.

    Args:
        original_imgs (torch.Tensor): _description_
        target_imgs (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    pass
