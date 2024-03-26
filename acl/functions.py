import torch
import numpy as np


def gaussian_loss_function(output, layers):
    N, D = output.shape  # batch size and single output size

    constant = 0.5 * D * N * np.log(np.pi)
    sum_squared_mappings = torch.square(output).sum() * 0.5

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


def narrow_gaussian(x, ell):
    return torch.exp(-0.5 * (x / ell) ** 2)


def approx_count_nonzero(x, ell=1e-3):
    # Approximation of || x ||_0
    return len(x) - narrow_gaussian(x, ell).sum(dim=-1)


def uniform_circle_loss_function_method2(output, layers):
    # N, D = output.shape  # batch size and single output size

    """Uniform density"""
    distances = torch.square(output)
    distances = torch.sum(distances, axis=1) ** 0.5 - 1
    clipped_distances = torch.max(torch.stack([distances, torch.zeros_like(distances)]), axis=0)[0]
    count = approx_count_nonzero(clipped_distances, 0.1)  # number of elements outside uniform distribution

    sum_uniform = -torch.sum(count * -10)  # negative log likelihood

    """Determinants"""
    temp = [torch.reshape(layers[f'layers.{i}'][0][1], (-1, 1)) for i in range(len(layers))]
    log_dets = torch.cat(temp, axis=1)
    sum_log_dets = torch.sum(log_dets)

    return sum_uniform - sum_log_dets


# n-gon uniform loss function

def compute_distances(data_points, sv, support_angles):
    # compute angles
    complex_tensor = data_points[:, 0] + data_points[:, 1] * torch.tensor(1j)
    point_angle = torch.angle(complex_tensor)
    # print(point_angle*360/2/np.pi)
    # get support vector pairs
    sv_pairs = sv[torch.searchsorted(support_angles, point_angle)], sv[
        torch.searchsorted(support_angles, point_angle) - 1]
    # get linear combinations to determine distance
    sv_pairs = torch.stack(sv_pairs, axis=1)
    sv_pairs = [svp.T for svp in sv_pairs]
    distances = torch.abs(torch.linalg.solve(torch.stack(sv_pairs), test_points)).sum(axis=1)
    # print(torch.linalg.solve(torch.stack(sv_pairs), test_points))
    return distances


def compute_ngon_sv_and_angles(n):
    support_angles = torch.linspace(0, 2 * torch.pi, n + 1)
    # construct support vectors
    sv = []
    for a in support_angles[:-1]:
        sv.append(torch.tensor([torch.cos(a), torch.sin(a)]))
    sv = torch.stack(sv)

    return sv, support_angles


def uniform_ngon_loss_function(output, layers, sv, support_angles):
    distances = compute_distances(output, sv, support_angles) - 1
    clipped_distances = torch.max(torch.stack([distances, torch.zeros_like(distances)]), axis=0)[0]
    count = approx_count_nonzero(clipped_distances, 0.1)  # number of elements outside uniform distrivution

    sum_uniform = -torch.sum(count * -10)  # negative log likelihood

    """Determinants"""
    temp = [torch.reshape(layers[f'layers.{i}'][0][1], (-1, 1)) for i in range(len(layers))]
    log_dets = torch.cat(temp, axis=1)
    sum_log_dets = torch.sum(log_dets)

    return sum_uniform - sum_log_dets
