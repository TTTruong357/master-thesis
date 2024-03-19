from turtle import forward
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from scipy import interpolate


class LeakySoftplus(nn.Module):
    def __init__(self, alpha: float = 0.1) -> None:
        super(LeakySoftplus, self).__init__()
        self.alpha = alpha

    def forward(self, input: Tensor) -> Tensor:
        softplus = torch.log1p(torch.exp(-torch.abs(input))) + torch.maximum(input, torch.tensor(0))
        output = self.alpha * input + (1 - self.alpha) * softplus
        return output


def lifted_sigmoid(x, alpha=0.1):
    """derivative of leaky softplus"""
    return alpha + (1 - alpha) * torch.sigmoid(x)


class InvertedLeakySoftplus(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super(InvertedLeakySoftplus, self).__init__()
        self.alpha = alpha

    def forward(self, input: Tensor):
        x = torch.arange(-1000., 1000.001, 0.001)
        activation = LeakySoftplus()
        y = activation(x)
        tck = interpolate.splrep(y, x, s=0)
        """data is first moved to cpu and then converted to numpy array"""
        yfit = interpolate.splev(input.cpu().detach().numpy(), tck, der=0)
        return torch.tensor(yfit, dtype=torch.float32)


class InvertedLeakySoftplus2(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, y):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # activation = LeakySoftplus()
        # x = activation(y)
        # y_approx = activation(x)
        # factor = 0.1
        # step = 0.1
        # while torch.any(torch.sum((y_approx - y)**2, axis=1) > 1e-6):
        #     x[torch.where(y_approx - y > 0)] = x[torch.where(y_approx - y > 0)]-step
        #     x[torch.where(y_approx - y < 0)] = x[torch.where(y_approx - y < 0)]+step
        #     y_approx = activation(x)
        #     #print(y_approx, x)
        #     step *= 0.99

        # activation = LeakySoftplus()
        # x = torch.arange(-1000., 1000.001, 0.001).to(device)
        # y_real = activation(x)
        # x = x[torch.searchsorted(y_real, y)]

        activation = LeakySoftplus()
        x = activation(y)
        y_approx = activation(x)

        step = torch.ones_like(y[:, 0])

        sign = torch.sign(torch.sum(y - y_approx, axis=1))
        sign_changed = torch.zeros_like(y[:, 0], dtype=torch.bool)
        first_change = torch.zeros_like(y[:, 0], dtype=torch.bool)

        x += ((y - y_approx).T * step).T

        while torch.any(torch.abs(torch.sum(y - y_approx, axis=1)) > 1e-6):
            y_approx = activation(x)

            new_sign = torch.sign(torch.sum(y - y_approx, axis=1))
            sign_changed = torch.tensor(sign * new_sign < 0)
            sign = new_sign

            first_change = first_change | sign_changed | (sign == 0.)
            step[torch.logical_not(first_change)] *= 2
            step[sign_changed] /= 2

            x += ((y - y_approx).T * step).T

        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x, = ctx.saved_tensors
        alpha = 0.1
        der = alpha + (1 - alpha) / (1 + torch.exp(x)) * torch.exp(x)
        return grad_output * 1 / der


class InvertedLLayer(nn.Module):
    def __init__(self, inverted_weight=None, inverted_bias=None) -> None:
        super(InvertedLLayer, self).__init__()
        self.inverted_weight = inverted_weight
        self.inverted_bias = inverted_bias

    def forward(self, input: Tensor, device="cuda:0") -> Tensor:
        input = input.to(device)
        y_tilde = torch.t(input - self.inverted_bias)
        x_tilde = torch.linalg.solve(self.inverted_weight, y_tilde)
        x = torch.t(x_tilde)
        return x


class InvertedULayer(nn.Module):
    def __init__(self, inverted_weight=None) -> None:
        super(InvertedULayer, self).__init__()
        self.inverted_weight = inverted_weight

    def forward(self, input: Tensor, device="cuda:0") -> Tensor:
        input = input.to(device)
        y_tilde = torch.t(input)
        x_tilde = torch.linalg.solve(self.inverted_weight, y_tilde)
        x = torch.t(x_tilde)
        return x


"""loss functions"""


def gaussian_log_likelihood(output, model, layers):
    """compute the log likelihood with change of variables formula, average per pixel"""
    N, D = output.shape  # batch size and single output size

    """First summand"""
    constant = torch.from_numpy(np.array(0.5 * D * N * np.log(np.pi))).type(torch.float64)

    """Second summand"""
    sum_squared_mappings = torch.square(output)
    sum_squared_mappings = torch.sum(sum_squared_mappings)
    sum_squared_mappings = 0.5 * sum_squared_mappings

    """Third summand"""
    """log diagonals of U"""
    log_diagonals_triu = []
    for param in model.parameters():
        if len(param.shape) == 2 and param[1, 0] == 0:  # if upper triangular and matrix
            log_diagonals_triu.append(torch.log(torch.abs(torch.diag(param))))

    log_derivatives = []
    for i in range((len(layers) - 1) * 2):
        """layers are outputs of the L-Layer"""
        """lifted sigmoid = derivative of leaky softplus"""
        if i % 2 != 0:
            log_derivatives.append(
                torch.log(torch.abs(lifted_sigmoid(layers["intermediate_lu_blocks.{}".format(i)][0]))))
    log_derivatives.append(torch.log(torch.abs(lifted_sigmoid(layers["final_lu_block.1"][0]))))

    """lu-blocks 1,...,M-1"""
    volume_corr = 0
    for l in range(len(log_diagonals_triu) - 1):
        summand = torch.zeros(N, D).to("cuda:0")
        summand = summand + log_derivatives[l]
        summand = summand + log_diagonals_triu[l]
        volume_corr = volume_corr + torch.sum(summand)

    """lu-block M"""
    last = log_diagonals_triu[len(log_diagonals_triu) - 1]
    last = N * torch.sum(last)

    output = constant + sum_squared_mappings - last - volume_corr
    return output


def uniform_circle_log_likelihood(output, model, layers, density_param):
    """compute the log likelihood with change of variables formula, average per pixel"""
    N, D = output.shape  # batch size and single output size

    """Uniform density"""
    dist = torch.square(output)
    dist = torch.sum(dist, axis=1) ** 0.5
    sum_uniform = -torch.sum(
        torch.log((1 / dist ** density_param[1]) ** (dist ** density_param[0])))  # negative log likelihood

    """Third summand"""
    """log diagonals of U"""
    log_diagonals_triu = []
    for param in model.parameters():
        if len(param.shape) == 2 and param[1, 0] == 0:  # if upper triangular and matrix
            log_diagonals_triu.append(torch.log(torch.abs(torch.diag(param))))

    log_derivatives = []
    for i in range((len(layers) - 1) * 2):
        """layers are outputs of the L-Layer"""
        """lifted sigmoid = derivative of leaky softplus"""
        if i % 2 != 0:
            log_derivatives.append(
                torch.log(torch.abs(lifted_sigmoid(layers["intermediate_lu_blocks.{}".format(i)][0]))))
    log_derivatives.append(torch.log(torch.abs(lifted_sigmoid(layers["final_lu_block.1"][0]))))

    """lu-blocks 1,...,M-1"""
    volume_corr = 0
    for l in range(len(log_diagonals_triu) - 1):
        summand = torch.zeros(N, D).to("cuda:0")
        summand = summand + log_derivatives[l]
        summand = summand + log_diagonals_triu[l]
        volume_corr = volume_corr + torch.sum(summand)

    """lu-block M"""
    last = log_diagonals_triu[len(log_diagonals_triu) - 1]
    last = N * torch.sum(last)

    return sum_uniform - last - volume_corr


def narrow_gaussian(x, ell):
    return torch.exp(-0.5 * (x / ell) ** 2)


def approx_count_nonzero(x, ell=1e-3):
    # Approximation of || x ||_0
    return len(x) - narrow_gaussian(x, ell).sum(dim=-1)


def uniform_circle_log_likelihood_method2(output, model, layers):
    """compute the log likelihood with change of variables formula, average per pixel"""
    N, D = output.shape  # batch size and single output size

    """Uniform density"""
    distances = torch.square(output)
    distances = torch.sum(distances, axis=1) ** 0.5 - 1
    clipped_distances = torch.max(torch.stack([distances, torch.zeros_like(distances)]), axis=0)[0]
    count = approx_count_nonzero(clipped_distances, 0.1)  # number of elements outside uniform distribution

    sum_uniform = -torch.sum(count * -10)  # negative log likelihood

    """Third summand"""
    """log diagonals of U"""
    log_diagonals_triu = []
    for param in model.parameters():
        if len(param.shape) == 2 and param[1, 0] == 0:  # if upper triangular and matrix
            log_diagonals_triu.append(torch.log(torch.abs(torch.diag(param))))

    log_derivatives = []
    for i in range((len(layers) - 1) * 2):
        """layers are outputs of the L-Layer"""
        """lifted sigmoid = derivative of leaky softplus"""
        if i % 2 != 0:
            log_derivatives.append(
                torch.log(torch.abs(lifted_sigmoid(layers["intermediate_lu_blocks.{}".format(i)][0]))))
    log_derivatives.append(torch.log(torch.abs(lifted_sigmoid(layers["final_lu_block.1"][0]))))

    """lu-blocks 1,...,M-1"""
    volume_corr = 0
    for l in range(len(log_diagonals_triu) - 1):
        summand = torch.zeros(N, D, device="cuda:0")
        summand = summand + log_derivatives[l]
        summand = summand + log_diagonals_triu[l]
        volume_corr = volume_corr + torch.sum(summand)

    """lu-block M"""
    last = log_diagonals_triu[len(log_diagonals_triu) - 1]
    last = N * torch.sum(last)

    return sum_uniform - last - volume_corr
