import torch
import torch.nn as nn
import numpy as np
from functools import partial


class CouplingLayer(nn.Module):

    def __init__(self, layer_size=2, split_size_x1=1, nn_size=(2, 100)):
        super(CouplingLayer, self).__init__()

        # random split x = (x1, x2)
        choice = np.random.choice(layer_size, split_size_x1, replace=False)
        ind = np.zeros(layer_size, dtype=bool)
        ind[choice] = True
        temp = np.arange(layer_size)
        self.split_index = (temp[ind], temp[~ind])

        self.t_net = nn.ModuleList()
        self.s_net = nn.ModuleList()

        self.activation = nn.ReLU()

        self.t_net.append(nn.Linear(split_size_x1, nn_size[1]))
        for _ in range(nn_size[0]):
            self.t_net.append(nn.Linear(nn_size[1], nn_size[1]))
        self.t_net.append(nn.Linear(nn_size[1], layer_size - split_size_x1))

        self.s_net.append(nn.Linear(split_size_x1, nn_size[1]))
        for _ in range(nn_size[0]):
            self.s_net.append(nn.Linear(nn_size[1], nn_size[1]))
        self.s_net.append(nn.Linear(nn_size[1], layer_size - split_size_x1))

    def forward(self, x, reverse=False):
        t = self.t_net[0](x[:, self.split_index[0]])
        t = self.activation(t)
        for i in range(1, len(self.t_net) - 1):
            t = self.t_net[i](t)
            if i % 2 == 0:
                t = self.activation(t)
            else:
                t = torch.sigmoid(t)
        t = self.t_net[-1](t)  # last layer no activation

        s = self.s_net[0](x[:, self.split_index[0]])
        s = self.activation(s)
        for i in range(1, len(self.s_net) - 1):
            s = self.s_net[i](s)
            if i % 2 == 0:
                s = self.activation(s)
            else:
                s = torch.sigmoid(s)
        s = self.s_net[-1](s)

        z = torch.zeros_like(x)

        if reverse:  # reverse == True: Generating
            z[:, self.split_index[1]] = torch.exp(-s) * (x[:, self.split_index[1]] - t)
            z[:, self.split_index[0]] = x[:, self.split_index[0]]
            # y = torch.cat([x1,y2], dim=1)
            log_det = torch.sum(-s, axis=1)
            # print(self.split_index[1], torch.exp(-s), - t)
        else:  # reverse == False: Normalizing
            z[:, self.split_index[1]] = torch.exp(s) * x[:, self.split_index[1]] + t
            z[:, self.split_index[0]] = x[:, self.split_index[0]]
            # y = torch.cat([x1,y2], dim=1)
            log_det = torch.sum(s, axis=1)
            # print(self.split_index[1], torch.exp(s), t)
        return z, log_det

    def join(self, x1, x2):
        pass


class NN(nn.Module):
    def __init__(self, num_coupling_layers=1, layer_size=2, split_size_x1=1, nn_size=(2, 100), device="cuda:0"):
        super(NN, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_coupling_layers):
            self.layers.append(CouplingLayer(layer_size, split_size_x1, nn_size))

    def forward(self, x, reverse=False):
        if reverse:
            for layer in reversed(self.layers):
                x, log_det = layer(x, reverse)
        else:
            for layer in self.layers:
                x, log_det = layer(x, reverse)
        return x


def save_activations(activations_dict, name, blu, bla, out):
    activations_dict[name].append(out)


def register_activation_hooks(model, layers_to_save):
    """register forward hooks in specified layers"""
    activations_dict = {name: [] for name in layers_to_save}
    for name, module in model.named_modules():
        if name in layers_to_save:
            module.register_forward_hook(partial(save_activations, activations_dict, name))
    return activations_dict
