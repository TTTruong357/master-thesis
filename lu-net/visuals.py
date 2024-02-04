import torch
from functions import lifted_sigmoid
from model import register_activation_hooks
import numpy as np

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import pyplot as plt


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
    last = torch.sum(last)

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
    #print(last, summand, sum_uniform)
    last = torch.sum(last)
    #summand = summand + last # equivalent, last determinant of last layer applies for all rows of batch

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


def plot_generating_uniform_density(model, model_inverted, device, density_param, train_loader, sampling_data, grid_width,
                                    x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), x_lim=(-1, 1), y_lim=(-1, 1),
                                    density_function=compute_uniform_circle_density):
    fig, ax = plt.subplots()

    plt.xlim(*x_lim)
    plt.ylim(*y_lim)

    x = np.linspace(*x_range, grid_width)
    y = np.linspace(*y_range, grid_width)

    xv, yv = np.meshgrid(x, y, indexing='xy')
    horizontal_lines = np.stack((xv, yv), axis=2)

    xv, yv = np.meshgrid(x, y, indexing='ij')
    vertical_lines = np.stack((xv, yv), axis=2)  # vertical_lines

    all_grid_points = np.concatenate(np.concatenate((horizontal_lines, vertical_lines)), axis=0)
    all_grid_points = torch.tensor(all_grid_points, dtype=torch.float32).to(device)

    t = model_inverted(all_grid_points).detach()
    t = t.to('cpu')
    t = t.numpy()
    x = np.array(t)[:, 0]
    y = np.array(t)[:, 1]
    z = density_function(model, torch.tensor(t, dtype=torch.float32).to(device),device, density_param)

    try:
        cntr2 = ax.tricontourf(x, y, z, levels=100, cmap="OrRd")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cntr2, cax=cax)
    except:
        pass

    # print(z)

    horizontal_transformed, vertical_transformed = np.split(np.array(t), 2)

    for h in np.split(horizontal_transformed, grid_width):
        ax.plot(h[:, 0], h[:, 1], c='b', linewidth=1.)

    for v in np.split(vertical_transformed, grid_width):
        ax.plot(v[:, 0], v[:, 1], c='b', linewidth=1.)

    ax.scatter(train_loader[:, 0][:3200], train_loader[:, 1][:3200], c='black', alpha=1, s=2)
    output = model_inverted(torch.tensor(sampling_data, dtype=torch.float32).to(device))
    output = output.to('cpu').detach().numpy()
    ax.scatter(np.array(output)[:, 0], np.array(output)[:, 1], c='green', alpha=1, s=2)

    return 0
