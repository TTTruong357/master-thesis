import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model import register_activation_hooks
import numpy as np
from matplotlib import pyplot as plt


def gaussian_likelihoods(data, model, layers):
    N, D = data.shape  # batch size and single output size
    # print(layers)
    """First summand"""
    constant = torch.from_numpy(np.array(0.5 * D * np.log(np.pi))).type(torch.float64)

    """Second summand"""
    sum_squared_mappings = torch.square(data)
    sum_squared_mappings = torch.sum(sum_squared_mappings, axis=1)
    sum_squared_mappings = 0.5 * sum_squared_mappings

    """Determinants"""
    log_dets = torch.cat([torch.reshape(layers[f'layers.{i}'][0][1], (-1, 1)) for i in range(len(layers))], axis=1)

    print(sum_squared_mappings.shape, log_dets.shape)

    output = constant + sum_squared_mappings - torch.sum(log_dets, axis=1)

    print(output, constant, sum_squared_mappings, torch.sum(log_dets, axis=1))

    output = output.to('cpu')

    return np.e ** np.array(-output)


def compute_gaussian_density(model, x_grid):
    layers = []
    for j in range(len(model.layers)):
        layers.append("layers.{}".format(j))

    with torch.no_grad():
        saved_layers = register_activation_hooks(model, layers_to_save=layers)
        data = x_grid
        # data = x_grid.clone().to(device)
        output = model(data, reverse=False)
        density = gaussian_likelihoods(output, model, saved_layers)

    return density


def uniform_circle_likelihoods(output, model, layers, device):
    uniform = torch.zeros(output.shape[0], device=device)
    dist = torch.square(output)
    dist = torch.sum(dist, axis=1) ** 0.5
    uniform[torch.where(dist > 1)] = 10

    """Determinants"""
    log_dets = torch.cat([torch.reshape(layers[f'layers.{i}'][0][1], (-1, 1)) for i in range(len(layers))], axis=1)

    output = uniform - torch.sum(log_dets, axis=1)
    output = output.to('cpu')
    return np.e ** np.array(-output)


def compute_uniform_circle_density(model, x_grid, device):
    layers = []
    for j in range(len(model.layers)):
        layers.append("layers.{}".format(j))

    with torch.no_grad():
        saved_layers = register_activation_hooks(model, layers_to_save=layers)
        data = x_grid
        # data = x_grid.clone().to(device)
        output = model(data, reverse=False)
        density = uniform_circle_likelihoods(output, model, saved_layers, device)

    return density


def plot_transformed_grid_and_density(model, train_loader, device, sampling_data, grid_width,
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
    all_grid_points = torch.tensor(all_grid_points, device=device)

    t = model(all_grid_points, reverse=True).detach()
    t = t.to('cpu')
    t = t.numpy()
    temp_t = t[:t.shape[0] // 2]
    x = np.array(temp_t)[:, 0]
    y = np.array(temp_t)[:, 1]
    z = density_function(model, torch.tensor(temp_t, device=device), device)

    cntr2 = ax.tricontourf(x, y, z, levels=100, cmap="OrRd")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cntr2, cax=cax)

    horizontal_transformed, vertical_transformed = np.split(np.array(t), 2)

    for h in np.split(horizontal_transformed, grid_width):
        ax.plot(h[:, 0], h[:, 1], c='b', linewidth=1.)

    for v in np.split(vertical_transformed, grid_width):
        ax.plot(v[:, 0], v[:, 1], c='b', linewidth=1.)

    ax.scatter(train_loader[:, 0][:3200], train_loader[:, 1][:3200], c='black', alpha=1, s=2)
    output = model(torch.tensor(sampling_data, device=device), reverse=True)
    output = output.to('cpu').detach().numpy()
    ax.scatter(np.array(output)[:, 0], np.array(output)[:, 1], c='green', alpha=1, s=2)

    return 0







"""Old version"""

# def uniform_circle_likelihoods(output, model, layers, density_param):
#     dist = torch.square(output)
#     dist = torch.sum(dist, axis=1) ** 0.5
#     sum_uniform = -torch.log((1 / dist ** density_param[1]) ** (dist ** density_param[0]))  # negative log likelihood
#
#     """Determinants"""
#     log_dets = torch.cat([torch.reshape(layers[f'layers.{i}'][0][1], (-1, 1)) for i in range(len(layers))], axis=1)
#
#     output = sum_uniform - torch.sum(log_dets, axis=1)
#
#     print(output, sum_uniform, torch.sum(log_dets, axis=1))
#
#     output = output.to('cpu')
#
#     return np.e ** np.array(-output)
#
#
# def compute_uniform_circle_density(model, x_grid, density_param):
#     layers = []
#     for j in range(len(model.layers)):
#         layers.append("layers.{}".format(j))
#
#     with torch.no_grad():
#         saved_layers = register_activation_hooks(model, layers_to_save=layers)
#         data = x_grid
#         # data = x_grid.clone().to(device)
#         output = model(data, reverse=False)
#         density = uniform_circle_likelihoods(output, model, saved_layers, density_param)
#
#     return density
#
#
# # sampling_data = generate_ellipse(1000, (1, 1))
#
#
# def plot_generating_uniform_density(model, device, density_param, train_loader, sampling_data, grid_width,
#                                     x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), x_lim=(-1, 1), y_lim=(-1, 1),
#                                     density_function=compute_uniform_circle_density):
#     fig, ax = plt.subplots()
#
#     plt.xlim(*x_lim)
#     plt.ylim(*y_lim)
#
#     x = np.linspace(*x_range, grid_width)
#     y = np.linspace(*y_range, grid_width)
#
#     xv, yv = np.meshgrid(x, y, indexing='xy')
#     horizontal_lines = np.stack((xv, yv), axis=2)
#
#     xv, yv = np.meshgrid(x, y, indexing='ij')
#     vertical_lines = np.stack((xv, yv), axis=2)  # vertical_lines
#
#     all_grid_points = np.concatenate(np.concatenate((horizontal_lines, vertical_lines)), axis=0)
#     all_grid_points = torch.tensor(all_grid_points).to(device)
#
#     t = model(all_grid_points, reverse=True).detach()
#     t = t.to('cpu')
#     t = t.numpy()
#     x = np.array(t)[:, 0]
#     y = np.array(t)[:, 1]
#     z = density_function(model, torch.tensor(t).to(device), density_param)
#
#     try:
#         cntr2 = ax.tricontourf(x, y, z, cmap="OrRd")
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         plt.colorbar(cntr2, cax=cax)
#     except:
#         pass
#
#     # print(z)
#
#     horizontal_transformed, vertical_transformed = np.split(np.array(t), 2)
#
#     for h in np.split(horizontal_transformed, grid_width):
#         ax.plot(h[:, 0], h[:, 1], c='b', linewidth=1.)
#
#     for v in np.split(vertical_transformed, grid_width):
#         ax.plot(v[:, 0], v[:, 1], c='b', linewidth=1.)
#
#     ax.scatter(train_loader[:, 0][:3200], train_loader[:, 1][:3200], c='black', alpha=1, s=2)
#     output = model(torch.tensor(sampling_data).to(device), reverse=True)
#     output = output.to('cpu').detach().numpy()
#     ax.scatter(np.array(output)[:, 0], np.array(output)[:, 1], c='green', alpha=1, s=2)
#
#     return 0
#
#
# def plot_generating_gaussian_density(model, device, density_param, train_loader, sampling_data, grid_width,
#                                     x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), x_lim=(-1, 1), y_lim=(-1, 1)):
#     fig, ax = plt.subplots()
#
#     plt.xlim(*x_lim)
#     plt.ylim(*y_lim)
#
#     x = np.linspace(*x_range, grid_width)
#     y = np.linspace(*y_range, grid_width)
#
#     xv, yv = np.meshgrid(x, y, indexing='xy')
#     horizontal_lines = np.stack((xv, yv), axis=2)
#
#     xv, yv = np.meshgrid(x, y, indexing='ij')
#     vertical_lines = np.stack((xv, yv), axis=2)  # vertical_lines
#
#     all_grid_points = np.concatenate(np.concatenate((horizontal_lines, vertical_lines)), axis=0)
#     all_grid_points = torch.tensor(all_grid_points).to(device)
#
#     t = model(all_grid_points, reverse=True).detach()
#     t = t.to('cpu')
#     t = t.numpy()
#     x = np.array(t)[:, 0]
#     y = np.array(t)[:, 1]
#     z = compute_gaussian_density(model, torch.tensor(t).to(device))
#
#     try:
#         cntr2 = ax.tricontourf(x, y, z, cmap="OrRd")
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         plt.colorbar(cntr2, cax=cax)
#     except:
#         pass
#
#     # print(z)
#
#     horizontal_transformed, vertical_transformed = np.split(np.array(t), 2)
#
#     for h in np.split(horizontal_transformed, grid_width):
#         ax.plot(h[:, 0], h[:, 1], c='b', linewidth=1.)
#
#     for v in np.split(vertical_transformed, grid_width):
#         ax.plot(v[:, 0], v[:, 1], c='b', linewidth=1.)
#
#     ax.scatter(train_loader[:, 0][:3200], train_loader[:, 1][:3200], c='black', alpha=1, s=2)
#     output = model(torch.tensor(sampling_data).to(device), reverse=True)
#     output = output.to('cpu').detach().numpy()
#     ax.scatter(np.array(output)[:, 0], np.array(output)[:, 1], c='green', alpha=1, s=2)
#
#     return 0