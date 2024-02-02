import torch
from functions import lifted_sigmoid
from model import register_activation_hooks
import numpy as np

def gaussian_likelihoods(output, model, layers):
    """compute the log likelihood with change of variables formula, average per pixel"""
    N, D = output.shape  # batch size and single output size

    """First summand"""
    constant = torch.from_numpy(np.array(0.5 * D * np.log(np.pi))).type(torch.float64)

    """Second summand"""
    sum_squared_mappings = torch.square(output)
    sum_squared_mappings = torch.sum(sum_squared_mappings, axis=1)
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
    summand = torch.zeros(N, D).to("cuda:0")
    for l in range(len(log_diagonals_triu) - 1):
        summand = summand + log_derivatives[l]
        summand = summand + log_diagonals_triu[l]

    """lu-block M"""
    last = log_diagonals_triu[len(log_diagonals_triu) - 1]
    last = torch.sum(last) # AXIS = 1 ?!?!?!?!?

    output = constant + sum_squared_mappings - last - summand.sum(axis=1)

    return np.e ** np.array(-output.to('cpu'))


def uniform_circle_likelihoods(output, model, layers, density_param):
    """compute the log likelihood with change of variables formula, average per pixel"""
    N, D = output.shape  # batch size and single output size

    #     """First summand"""
    #     constant = torch.from_numpy(np.array(0.5 * D * np.log(np.pi))).type(torch.float64)

    #     """Second summand"""
    #     sum_squared_mappings = torch.square(output)
    #     sum_squared_mappings = torch.sum(sum_squared_mappings, axis=1)
    #     sum_squared_mappings = 0.5 * sum_squared_mappings

    dist = torch.square(output)
    dist = torch.sum(dist, axis=1) ** 0.5
    sum_uniform = -torch.log((1 / dist ** density_param[1]) ** (dist ** density_param[0]))  # negative log likelihood

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
    summand = torch.zeros(N, D).to("cuda:0")
    for l in range(len(log_diagonals_triu) - 1):
        summand = summand + log_derivatives[l]
        summand = summand + log_diagonals_triu[l]

    """lu-block M"""
    last = log_diagonals_triu[len(log_diagonals_triu) - 1]
    last = torch.sum(last) # AXIS = 1 ?!?!?

    output = sum_uniform - last - summand.sum(axis=1)

    return np.e ** np.array(-output.to('cpu'))


def compute_uniform_circle_density(model, x_grid, device, density_param):
    """outputs of L-layer are needed for the loss function"""
    layers = []
    for j in range(len(model.intermediate_lu_blocks)):
        if j % 2 != 0:
            layers.append("intermediate_lu_blocks.{}".format(j))
    layers.append("final_lu_block.1")


    with torch.no_grad():
        saved_layers = register_activation_hooks(model, layers_to_save=layers)
        data = x_grid.clone().to(device)
        output = model(data)
        density = uniform_circle_likelihoods(output, model, saved_layers, density_param)

    return density


def compute_gaussian_density(model, x_grid, device):
    """outputs of L-layer are needed for the loss function"""
    layers = []
    for j in range(len(model.intermediate_lu_blocks)):
        if j % 2 != 0:
            layers.append("intermediate_lu_blocks.{}".format(j))
    layers.append("final_lu_block.1")


    with torch.no_grad():
        saved_layers = register_activation_hooks(model, layers_to_save=layers)
        data = x_grid.clone().to(device)
        output = model(data)
        density = gaussian_likelihoods(output, model, saved_layers)

    return density


