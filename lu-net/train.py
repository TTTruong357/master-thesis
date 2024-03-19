import torch
from functions import gaussian_log_likelihood, uniform_circle_log_likelihood, uniform_circle_log_likelihood_method2
from tqdm import tqdm
#from visuals import add_batch_loss
from torch import nn
from model import register_activation_hooks

"""training of gaussian data"""


def training_routine_gaussian(model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    train_loss = 0
    i = 1

    """outputs of L-layer are needed for the loss function"""
    layers = []
    for j in range(len(model.intermediate_lu_blocks)):
        if j % 2 != 0:
            layers.append("intermediate_lu_blocks.{}".format(j))
    layers.append("final_lu_block.1")

    for k in tqdm(range(int(len(train_loader) / batch_size))):
        saved_layers = register_activation_hooks(model, layers_to_save=layers)
        inputs = train_loader[k * batch_size: k * batch_size + batch_size]
        inputs = inputs.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = gaussian_log_likelihood(output, model, saved_layers)
        train_loss += loss
        _, layer_size = output.shape
        #add_batch_loss(epoch, i, loss.item() / (batch_size * layer_size))
        #print("batch loss: " + str(loss.item() / (batch_size * layer_size)))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=1)
        optimizer.step()
        i += 1


def training_routine_uniform_method2(model, device, train_loader, optimizer, epoch, batch_size,
                                     loss_function=uniform_circle_log_likelihood_method2, grid_penalty=False,
                                     grid_input=None, penalty_weight=1e-3):
    model.train()
    train_loss = 0
    i = 1

    """outputs of L-layer are needed for the loss function"""
    layers = []
    for j in range(len(model.intermediate_lu_blocks)):
        if j % 2 != 0:
            layers.append("intermediate_lu_blocks.{}".format(j))
    layers.append("final_lu_block.1")

    for k in tqdm(range(int(len(train_loader) / batch_size))):
        optimizer.zero_grad(set_to_none=True)
        loss = 0
        ###
        # SECOND GRID INPUT AND ADD TO LOSS FUNCTION
        if grid_penalty:
            grid_output = model(grid_input, reverse=True)
            loss += penalty_weight * torch.sum((grid_output - grid_input) ** 2)
            print(loss)
        ###
        saved_layers = register_activation_hooks(model, layers_to_save=layers)
        inputs = train_loader[k * batch_size: k * batch_size + batch_size]
        inputs = inputs.to(device)
        output = model(inputs)
        loss += loss_function(output, model, saved_layers)
        train_loss += loss
        # _, layer_size = output.shape
        # add_batch_loss(epoch, i, loss.item() / (batch_size * layer_size))
        # print("batch loss: " + str(loss.item() / (batch_size * layer_size)))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=1)
        optimizer.step()
        i += 1


def training_routine_uniform(model, device, train_loader, optimizer, epoch, batch_size,
                             loss_function=uniform_circle_log_likelihood, density_param=(5, 0.2)):
    model.train()
    train_loss = 0
    i = 1

    """outputs of L-layer are needed for the loss function"""
    layers = []
    for j in range(len(model.intermediate_lu_blocks)):
        if j % 2 != 0:
            layers.append("intermediate_lu_blocks.{}".format(j))
    layers.append("final_lu_block.1")

    for k in tqdm(range(int(len(train_loader) / batch_size))):
        saved_layers = register_activation_hooks(model, layers_to_save=layers)
        inputs = train_loader[k * batch_size: k * batch_size + batch_size]
        inputs = inputs.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = loss_function(output, model, saved_layers, density_param)
        train_loss += loss
        _, layer_size = output.shape
        #add_batch_loss(epoch, i, loss.item() / (batch_size * layer_size))
        #print("batch loss: " + str(loss.item() / (batch_size * layer_size)))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=1)
        optimizer.step()
        i += 1
