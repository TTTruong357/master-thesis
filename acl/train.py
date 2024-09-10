from functions import uniform_circle_loss_function_method2, get_grad_vector, isoperimetric_loss, orthogonal_projection, get_shapes, get_weights_from_vector, \
    assign_grad_weights, compute_exact_volume_change
from tqdm import tqdm
from model import register_activation_hooks, remove_activation_hooks
from torch import nn
import torch

# training routine for likelihood based training

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


# training routine for isoperimetric approach

def training_routine_isoperimetric(model, device, outer_points, inner_points, optimizer, method=0, penalty_weight=0.1):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    # train_loss = 0

    if method == 0:  # orthogonal projection

        """outputs of individual affine coupling layers are needed for the loss function"""
        layers = []
        for j in range(len(model.layers)):
            layers.append("layers.{}".format(j))

        saved_layers, handles = register_activation_hooks(model, layers_to_save=layers)

        # get volume grad
        optimizer.zero_grad(set_to_none=True)
        output1 = model(inner_points)
        volume = compute_exact_volume_change(saved_layers)
        volume.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=1)
        volume_grad = get_grad_vector(model, device)
        # print(volume, volume_grad[:10])
        print('volume', volume)
        output1.detach()
        volume.detach()

        # get loss grad
        optimizer.zero_grad(set_to_none=True)
        output2 = model(outer_points)
        boundary_loss = isoperimetric_loss(output2)
        boundary_loss.backward()  #
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=1)

        # project on subspace
        grad = get_grad_vector(model, device)
        grad_projected = orthogonal_projection(volume_grad, grad)
        shapes = get_shapes(model)
        grad_weigths = get_weights_from_vector(grad_projected, shapes)

        # assign values
        assign_grad_weights(model, grad_weigths)

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=1)

        # train step
        optimizer.step()
        volume = volume.detach()
        boundary_loss.detach()
        output1.detach()
        output2.detach()
        remove_activation_hooks(handles)

        return volume, boundary_loss

    else:  # additional loss term

        """outputs of individual affine coupling layers are needed for the loss function"""
        layers = []
        for j in range(len(model.layers)):
            layers.append("layers.{}".format(j))

        saved_layers, handles = register_activation_hooks(model, layers_to_save=layers)

        # get loss grad
        optimizer.zero_grad(set_to_none=True)

        output = model(torch.concat((outer_points, inner_points)))
        volume = compute_exact_volume_change(saved_layers)
        print('volume', volume)
        volume_diff = (volume - 1) ** 2

        boundary_loss = isoperimetric_loss(output[:outer_points.shape[0]])
        loss = (1 - penalty_weight) * boundary_loss + penalty_weight * volume_diff
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=1)

        # train step
        optimizer.step()
        print('boundary_loss', boundary_loss)
        volume = volume.detach()
        boundary_loss.detach()
        output.detach()
        remove_activation_hooks(handles)

        return volume, boundary_loss