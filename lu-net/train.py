import torch
from functions import gaussian_loss_function, uniform_circle_log_likelihood, uniform_circle_log_likelihood_method2
from tqdm import tqdm
# from visuals import add_batch_loss
from torch import nn
from model import register_activation_hooks, remove_activation_hooks

"""training of gaussian data"""


def training_routine_gaussian(model, train_loader, optimizer):
    model.train()
    train_loss = 0

    """outputs of L-layer are needed for the loss function"""
    layers = []
    for j in range(len(model.intermediate_lu_blocks)):
        if j % 2 != 0:
            layers.append("intermediate_lu_blocks.{}".format(j))
    layers.append("final_lu_block.1")

    for i, inputs in tqdm(enumerate(train_loader)):
        saved_layers, handles = register_activation_hooks(model, layers_to_save=layers)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = gaussian_loss_function(output, model, saved_layers)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=1)
        optimizer.step()
        loss = loss.detach()
        train_loss += loss
        remove_activation_hooks(handles)


def training_routine_uniform_method2(model, train_loader, optimizer,
                                     loss_function=uniform_circle_log_likelihood_method2, grid_penalty=False,
                                     grid_input=None, penalty_weight=1e-3):
    model.train()
    train_loss = 0

    """outputs of L-layer are needed for the loss function"""
    layers = []
    for j in range(len(model.intermediate_lu_blocks)):
        if j % 2 != 0:
            layers.append("intermediate_lu_blocks.{}".format(j))
    layers.append("final_lu_block.1")

    for i, input in tqdm(enumerate(train_loader)):
        optimizer.zero_grad(set_to_none=True)
        loss = 0
        ###
        # SECOND GRID INPUT AND ADD TO LOSS FUNCTION
        if grid_penalty:
            grid_output = model(grid_input, reverse=True)
            loss += penalty_weight * torch.sum((grid_output - grid_input) ** 2)
            print(loss)
        ###
        saved_layers, handles = register_activation_hooks(model, layers_to_save=layers)
        output = model(input)
        loss += loss_function(output, model, saved_layers)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=1)
        optimizer.step()
        loss = loss.detach()
        train_loss += loss
        remove_activation_hooks(handles)


def training_routine_uniform(model, train_loader, optimizer,
                             loss_function=uniform_circle_log_likelihood, density_param=(5, 0.2)):
    model.train()
    train_loss = 0

    """outputs of L-layer are needed for the loss function"""
    layers = []
    for j in range(len(model.intermediate_lu_blocks)):
        if j % 2 != 0:
            layers.append("intermediate_lu_blocks.{}".format(j))
    layers.append("final_lu_block.1")

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
