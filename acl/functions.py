import torch
import numpy as np


def gaussian_loss_function(output, model, layers):
    N, D = output.shape  # batch size and single output size
    # print(layers)
    """First summand"""
    constant = torch.from_numpy(np.array(0.5 * D * N * np.log(np.pi))).type(torch.float32)

    """Second summand"""
    sum_squared_mappings = torch.square(output)
    sum_squared_mappings = torch.sum(sum_squared_mappings)
    sum_squared_mappings = 0.5 * sum_squared_mappings

    """Determinants"""
    temp = [torch.reshape(layers[f'layers.{i}'][0][1], (-1, 1)) for i in range(len(layers))]
    log_dets = torch.cat(temp, axis=1)
    sum_log_dets = torch.sum(log_dets)

    return constant + sum_squared_mappings - sum_log_dets


def uniform_circle_loss_function(output, model, layers, density_param):
    N, D = output.shape  # batch size and single output size

    """Uniform density"""
    dist = torch.square(output)
    dist = torch.sum(dist, axis=1) ** 0.5
    sum_uniform = -torch.sum(
        torch.log((1 / dist ** density_param[1]) ** (dist ** density_param[0])))  # negative log likelihood

    """Determinants"""
    temp = [torch.reshape(layers[f'layers.{i}'][0][1], (-1, 1)) for i in range(len(layers))]
    log_dets = torch.cat(temp, axis=1)
    sum_log_dets = torch.sum(log_dets)

    return sum_uniform - sum_log_dets
