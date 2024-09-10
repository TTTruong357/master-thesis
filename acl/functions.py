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
    """Failed first attempt"""

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


# Construction of Uniform density with augmented tail

def narrow_gaussian(x, ell):
    return torch.exp(-0.5 * (x / ell) ** 2)


def approx_count_nonzero(x, ell=1e-3):
    # Approximation of || x ||_0
    return len(x) - narrow_gaussian(x, ell).sum(dim=-1)


def uniform_circle_loss_function_method2(output, layers):

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


# Isoperimetrisches Problem

def isoperimetric_loss(points, grid_input=None):
    diff = torch.diff(torch.concatenate((points, points[:1])), axis=0)
    diff = torch.diff(points, axis=0)
    distances = torch.sum(diff ** 2, axis=1)

    boundary_loss = torch.sum(distances)

    print(f' boundary_loss: {boundary_loss}')

    return boundary_loss


def orthogonal_projection(v, grad):
    return grad - grad @ v * v / torch.norm(v) ** 2


def compute_exact_volume_change(layers):
    '''assuming volume of figure is 1'''
    temp = [torch.reshape(layers[f'layers.{i}'][0][1], (-1, 1)) for i in range(len(layers))]
    log_dets = torch.cat(temp, axis=1)
    return (torch.e ** log_dets.sum(axis=1)).mean()


def get_weights_vector(model):
    with torch.no_grad():
        weights = []

        for coupling_layer in model.layers:
            for linear_layer in coupling_layer.t_net:
                weights.append(linear_layer.weight)
                weights.append(linear_layer.bias)
            for linear_layer in coupling_layer.s_net:
                weights.append(linear_layer.weight)
                weights.append(linear_layer.bias)

        vector = torch.concatenate([w.flatten() for w in weights])
        return vector


def get_grad_vector(model, device):
    shapes = get_shapes(model)

    weights = []

    counter = 0
    for coupling_layer in model.layers:
        for linear_layer in coupling_layer.t_net:

            grad = linear_layer.weight.grad
            if grad is None:
                weights.append(torch.zeros(shapes[counter], device=device))
            else:
                weights.append(grad)
            counter += 1

            grad = linear_layer.bias.grad
            if grad is None:
                weights.append(torch.zeros(shapes[counter], device=device))
            else:
                weights.append(grad)
            counter += 1

        for linear_layer in coupling_layer.s_net:
            grad = linear_layer.weight.grad
            if grad is None:
                weights.append(torch.zeros(shapes[counter], device=device))
            else:
                weights.append(grad)
            counter += 1

            grad = linear_layer.bias.grad
            if grad is None:
                weights.append(torch.zeros(shapes[counter], device=device))
            else:
                weights.append(grad)
            counter += 1

    vector = torch.concatenate([w.flatten() for w in weights])
    return vector


def get_shapes(model):
    shapes = []

    for coupling_layer in model.layers:
        for linear_layer in coupling_layer.t_net:
            shapes.append(linear_layer.weight.shape)
            shapes.append(linear_layer.bias.shape)
        for linear_layer in coupling_layer.s_net:
            shapes.append(linear_layer.weight.shape)
            shapes.append(linear_layer.bias.shape)

    return shapes


def get_weights_from_vector(vector, shapes):
    lengths = [np.prod(s) for s in shapes]
    return [t.reshape(shapes[i]) for i, t in enumerate(torch.split(vector, lengths))]


def assign_grad_weights(model, grad_weights):
    counter = 0

    for coupling_layer in model.layers:
        for linear_layer in coupling_layer.t_net:
            linear_layer.weight.grad = grad_weights[counter]
            counter += 1
            linear_layer.bias.grad = grad_weights[counter]
            counter += 1
        for linear_layer in coupling_layer.s_net:
            linear_layer.weight.grad = grad_weights[counter]
            counter += 1
            linear_layer.bias.grad = grad_weights[counter]
            counter += 1
