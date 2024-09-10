import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model import register_activation_hooks, remove_activation_hooks
import numpy as np
from matplotlib import pyplot as plt


def gaussian_likelihoods(data, model, layers, device):
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

    output = constant + sum_squared_mappings - torch.sum(log_dets, axis=1)
    output = output.to('cpu')

    return np.e ** np.array(-output)


def compute_gaussian_density(model, x_grid, device):
    layers = []
    for j in range(len(model.layers)):
        layers.append("layers.{}".format(j))

    with torch.no_grad():
        saved_layers, handles = register_activation_hooks(model, layers_to_save=layers)
        data = x_grid
        output = model(data, reverse=False)
        density = gaussian_likelihoods(output, model, saved_layers, device)
        remove_activation_hooks(handles)
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
        saved_layers, handles = register_activation_hooks(model, layers_to_save=layers)
        data = x_grid
        output = model(data, reverse=False)
        density = uniform_circle_likelihoods(output, model, saved_layers, device)
        remove_activation_hooks(handles)
    return density


# Plot for Uniform loss function

def plot_transformed_circle_grid_and_density(model, train_loader, device, sampling_data, grid_shape,
                                             x_lim=(-1, 1), y_lim=(-1, 1),
                                             density_function=compute_uniform_circle_density, name='default.png'):
    fig, ax = plt.subplots()

    plt.xlim(*x_lim)
    plt.ylim(*y_lim)

    x = np.linspace(-1, 1, 300)
    y = np.linspace(-1, 1, 300)

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

    angles = np.linspace(0, 2 * np.pi, grid_shape[0])
    circles = [np.stack((np.cos(angles), np.sin(angles)), axis=1) * s for s in np.linspace(0.1, 1, grid_shape[1])]
    all_grid_points = np.concatenate(circles)
    all_grid_points = torch.tensor(all_grid_points, device=device)

    t = model(all_grid_points, reverse=True).detach()
    t = t.to('cpu')
    t = t.numpy()

    circles_transformed = np.split(np.array(t), grid_shape[1])
    for circle in circles_transformed:
        ax.plot(circle[:, 0], circle[:, 1], '-o', markersize=2, c='orange')

    ax.scatter(train_loader[:, 0][:3200], train_loader[:, 1][:3200], c='black', alpha=1, s=2)
    output = model(torch.tensor(sampling_data, device=device), reverse=True)
    output = output.to('cpu').detach().numpy()
    ax.scatter(np.array(output)[:, 0], np.array(output)[:, 1], c='green', alpha=1, s=2)

    plt.savefig(name)

    return 0


# Plot for Gaussian loss function

def plot_transformed_grid_and_density(model, train_loader, device, sampling_data, grid_width,
                                      x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), x_lim=(-1, 1), y_lim=(-1, 1),
                                      density_function=compute_uniform_circle_density, name='random.png'):
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
        ax.plot(h[:, 0], h[:, 1], c='b', linewidth=.1)

    for v in np.split(vertical_transformed, grid_width):
        ax.plot(v[:, 0], v[:, 1], c='b', linewidth=.1)

    ax.scatter(train_loader[:3200, 0], train_loader[:3200, 1], c='black', alpha=1, s=0.2)
    output = model(torch.tensor(sampling_data, device=device), reverse=True)
    output = output.to('cpu').detach().numpy()
    ax.scatter(np.array(output)[:, 0], np.array(output)[:, 1], c='green', alpha=1, s=0.2)

    plt.savefig(name)
    return 0


def plot_concat_density(device, model_circle, model_shape, train_loader, sigma_range=4, grid_width=30,
                        boundary_points=2000,
                        x_lim=(-1, 1), y_lim=(-1, 1), name='random.png'):
    x_range = (-sigma_range, sigma_range)
    y_range = (-sigma_range, sigma_range)

    x = np.linspace(*x_range, grid_width)
    y = np.linspace(*y_range, grid_width)

    xv, yv = np.meshgrid(x, y, indexing='xy')
    horizontal_lines = np.stack((xv, yv), axis=2)

    xv, yv = np.meshgrid(x, y, indexing='ij')
    vertical_lines = np.stack((xv, yv), axis=2)  # vertical_lines

    all_grid_points = np.concatenate(np.concatenate((horizontal_lines, vertical_lines)), axis=0)
    all_grid_points = torch.tensor(all_grid_points, device=device)

    t = model_circle(all_grid_points, reverse=True).detach()
    t = t.to('cpu')
    t = t.numpy()
    temp_t = t[:t.shape[0] // 2]
    z = compute_gaussian_density(model_circle, torch.tensor(temp_t, device=device), device)

    mask = np.ones_like(z)
    dist = np.square(temp_t)
    dist = np.sum(dist, axis=1) ** 0.5
    mask[np.where(dist > 1)] = 0
    z_circle = 1 / z * mask

    t = model_shape(all_grid_points, reverse=True).detach()
    t = t.to('cpu')
    t = t.numpy()
    temp_t = t[:t.shape[0] // 2]
    x = np.array(temp_t)[:, 0]
    y = np.array(temp_t)[:, 1]
    z_shape = compute_gaussian_density(model_shape, torch.tensor(temp_t, device=device), device)

    z = z_shape * z_circle

    z = np.nan_to_num(z, 0)

    fig, ax = plt.subplots()
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)

    cntr2 = ax.tricontourf(x, y, z, levels=100, cmap="OrRd")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cntr2, cax=cax)

    horizontal_transformed, vertical_transformed = np.split(np.array(t), 2)

    for h in np.split(horizontal_transformed, grid_width):
        ax.plot(h[:, 0], h[:, 1], c='b', linewidth=.1)

    for v in np.split(vertical_transformed, grid_width):
        ax.plot(v[:, 0], v[:, 1], c='b', linewidth=.1)

    angles = np.linspace(0, 2 * np.pi, boundary_points)
    circle = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    boundary = model_shape(model_circle(torch.tensor(circle, device=device)), reverse=True).detach().cpu().numpy()
    ax.scatter(np.array(boundary)[:, 0], np.array(boundary)[:, 1], c='green', alpha=1, s=2)

    ax.scatter(train_loader[:3200, 0], train_loader[:3200, 1], c='black', alpha=1, s=0.2)

    plt.savefig(name)


# Plot function for Isoperimetric Problem

def plot_transformed_shape_and_grid(model, device, original_shape, grid_width,
                                    x_lim=(-1, 1), y_lim=(-1, 1),
                                    density_function=compute_uniform_circle_density, name='default.png'):
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.xlim(*x_lim)
    plt.ylim(*y_lim)

    x = np.linspace(-5, 5, grid_width)
    y = np.linspace(-5, 5, grid_width)

    xv, yv = np.meshgrid(x, y, indexing='xy')
    horizontal_lines = np.stack((xv, yv), axis=2)

    xv, yv = np.meshgrid(x, y, indexing='ij')
    vertical_lines = np.stack((xv, yv), axis=2)  # vertical_lines

    all_grid_points = np.concatenate(np.concatenate((horizontal_lines, vertical_lines)), axis=0)
    all_grid_points = torch.tensor(all_grid_points, device=device, dtype=torch.float32)

    t = model(all_grid_points).detach()
    t = t.to('cpu')
    t = t.numpy()

    horizontal_transformed, vertical_transformed = np.split(np.array(t), 2)

    for h in np.split(horizontal_transformed, grid_width):
        ax.plot(h[:, 0], h[:, 1], c='b', linewidth=.1)

    for v in np.split(vertical_transformed, grid_width):
        ax.plot(v[:, 0], v[:, 1], c='b', linewidth=.1)

    output = model(torch.tensor(original_shape, device=device))
    output = output.to('cpu').detach().numpy()
    ax.plot(np.array(output)[:, 0], np.array(output)[:, 1], c='green', alpha=1, linestyle='-')

    plt.savefig(name)
