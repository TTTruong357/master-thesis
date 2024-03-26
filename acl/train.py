from functions import gaussian_loss_function, uniform_circle_loss_function_method2, uniform_circle_loss_function
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
    i = 1

    """outputs of individual affine coupling layers are needed for the loss function"""
    layers = []
    for j in range(len(model.layers)):
        layers.append("layers.{}".format(j))

    for k in range(int(len(train_loader) / batch_size)):
        saved_layers = register_activation_hooks(model, layers_to_save=layers)
        inputs = train_loader[k * batch_size: k * batch_size + batch_size]
        # inputs = inputs.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = loss_function(output, model, saved_layers, density_param)
        train_loss += loss
        # _, layer_size = output.shape
        # add_batch_loss(epoch, i, loss.item() / (batch_size * layer_size))
        # print("batch loss: " + str(loss.item() / (batch_size * layer_size)))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=1)
        optimizer.step()
        i += 1


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
        for k in tqdm(range(int(len(test_loader) / batch_size))):
            saved_layers = register_activation_hooks(model, layers_to_save=layers)
            inputs = test_loader[k * batch_size: k * batch_size + batch_size]
            # inputs = inputs.to(device)
            output = model(inputs)
            loss = loss_function(output, model, saved_layers, density_param)
            test_loss += loss

    return test_loss
