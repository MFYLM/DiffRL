from typing import List

import torch
import torch.nn as nn


class FlowMatching(torch.nn.Module):
    def __init__(self, model, forward_model):
        super(FlowMatching, self).__init__()
        print('initializing neural interpolant')
        self.model = model
        self.forward_model = forward_model

    def neural_interpolant(self, x0, x1, t0, t1):
        # given a set of x0's and x1's determines the linear interpolant between them

        # get our two path estimation
        xt = (1-t0) * x0 + t0 * x1 + (1/4 - (0.5 - t0)**2) * \
            self.forward_model(torch.concat([x0, x1], axis=1), t0.squeeze())
        xtp1 = (1-t1) * x0 + t1 * x1 + (1/4 - (0.5 - t1)**2) * \
            self.forward_model(torch.concat([x0, x1], axis=1), t1.squeeze())

        # estimate the gradient
        gr = (xtp1-xt)/(t1-t0)

        return xt, gr

    def forward(self, x0, x1):

        t = torch.rand(x0.shape[0]).to('cuda')
        xt, gr = self.neural_interpolant(x0, x1, t[:, None], t[:, None] + 1e-5)
        gr_hat = self.model(xt, t)

        return torch.nn.MSELoss()(gr, gr_hat)


class EmpiricalFlowMatching(FlowMatching):
    def __init__(self, model, forward_model, forward_noise_model):
        super(FlowMatching, self).__init__()
        print('initializing empirical neural interpolant')
        self.model = model
        self.forward_model = forward_model
        self.forward_noise_model = forward_noise_model

    def path_sampling(self, x0, x1, t0, t1, n_steps=25):
        # given a set of x0's and x1's determines the linear interpolant between them
        h = (t1-t0) / n_steps
        t = t0
        z = x0

        zs = [z]
        ts = [t]

        for i_step in range(n_steps):
            # perterb
            z = z + h * (self.forward_model(torch.concat([x0, x1], axis=1), t.squeeze()) + x1 - x0) + \
                (h**2) * torch.rand_like(x0) * \
                self.forward_noise_model(
                    torch.concat([x0, x1], axis=1), t.squeeze())
            # used to make z converge to x1
            if t.mean() > 0.7:
                z = z * (1-t) + x1 * t

            t = t + h

            zs += [z]
            ts += [t]

        return zs, ts

    def forward(self, x0, x1):

        B = x0.shape[0]

        t0 = torch.zeros(B).to(x0.device)[:, None]
        t1 = torch.ones(B).to(x1.device)[:, None]

        samples, ts = self.path_sampling(x0, x1, t0, t1)
        # take gradients
        gr = [(i-j)/(t_end-t_start) for i, j, t_start,
              t_end in zip(samples[1:], samples[:-1], ts[:-1], ts[1:])]

        gr_hat = [self.model(xt, time.squeeze())
                  for xt, time in zip(samples[:-1], ts[:-1])]

        gr = torch.stack(gr, axis=0)
        gr_hat = torch.stack(gr_hat, axis=0)

        return torch.nn.MSELoss()(gr, gr_hat)
