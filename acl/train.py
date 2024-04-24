from functions import gaussian_loss_function, uniform_circle_loss_function_method2, uniform_circle_loss_function, \
    compute_area, get_grad_vector, isoperimetric_loss, orthogonal_projection, get_shapes, get_weights_from_vector, \
    assign_grad_weights
from tqdm import tqdm
from model import register_activation_hooks, remove_activation_hooks
from torch import nn
import torch

"""training of gaussian data"""


def training_routine(model, train_loader, optimizer, loss_function):
    model.train()
    train_loss = 0

    layers = ["layers.{}".format(j) for j in range(len(model.layers))]

    for i, input in tqdm(enumerate(train_loader)):
        saved_layers, handles = register_activation_hooks(model, layers_to_save=layers)
        optimizer.zero_grad(set_to_none=True)
        output = model(input)
        loss = loss_function(output, saved_layers)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=1)
        optimizer.step()
        loss = loss.detach()
        train_loss += loss
        remove_activation_hooks(handles)


def training_routine_with_density_param(model, device, train_loader, optimizer, epoch, batch_size,
                                        loss_function=uniform_circle_loss_function,
                                        density_param=(5, 0.2)):
    model.train()
    train_loss = 0

    """outputs of individual affine coupling layers are needed for the loss function"""
    layers = []
    for j in range(len(model.layers)):
        layers.append("layers.{}".format(j))

    for i, input in tqdm(enumerate(train_loader)):
        saved_layers, handles = register_activation_hooks(model, layers_to_save=layers)
        optimizer.zero_grad(set_to_none=True)
        output = model(input)
        loss = loss_function(output, model, saved_layers, density_param)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=1)
        optimizer.step()
        loss = loss.detach()
        train_loss += loss
        remove_activation_hooks(handles)


def testing_routine(model, test_loader, loss_function=uniform_circle_loss_function_method2):
    model.eval()
    test_loss = 0

    """outputs of individual affine coupling layers are needed for the loss function"""
    layers = ["layers.{}".format(j) for j in range(len(model.layers))]

    with torch.no_grad():
        for i, input in tqdm(enumerate(test_loader)):
            saved_layers, handles = register_activation_hooks(model, layers_to_save=layers)
            output = model(input)
            loss = loss_function(output, saved_layers)
            test_loss += loss
            remove_activation_hooks(handles)

    return test_loss


def testing_routine_with_density_param(model, device, test_loader, batch_size,
                                       loss_function=uniform_circle_loss_function,
                                       density_param=(5, 0.2)):
    model.eval()
    test_loss = 0

    """outputs of individual affine coupling layers are needed for the loss function"""
    layers = []
    for j in range(len(model.layers)):
        layers.append("layers.{}".format(j))

    with torch.no_grad():
        for i, input in tqdm(enumerate(test_loader)):
            saved_layers, handles = register_activation_hooks(model, layers_to_save=layers)
            output = model(input)
            loss = loss_function(output, model, saved_layers, density_param)
            test_loss += loss
            remove_activation_hooks(handles)

    return test_loss


def training_routine_isoperimetric(model, train_data, optimizer):
    torch.autograd.set_detect_anomaly(False)
    model.train()
    # train_loss = 0

    # get volume grad
    optimizer.zero_grad(set_to_none=True)
    output = model(train_data)
    origin = output[:1]
    volume = compute_area(output[1:], origin)
    volume.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=1)
    volume_grad = get_grad_vector(model)
    # print(volume, volume_grad[:10])
    print('volume', volume)

    # get loss grad
    optimizer.zero_grad(set_to_none=True)
    output = model(train_data[1:])
    loss = isoperimetric_loss(output)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=1)

    # project on subspace
    grad = get_grad_vector(model)
    # print(grad[:10])
    grad_projected = orthogonal_projection(volume_grad, grad)
    # print(grad_projected[:10])
    shapes = get_shapes(model)
    grad_weigths = get_weights_from_vector(grad_projected, shapes)

    # assign values
    assign_grad_weights(model, grad_weigths)

    # train step
    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=1)
    optimizer.step()
    loss = loss.detach()
    print('loss', loss)
    return volume_grad, grad