import torch
import torch.nn as nn
import numpy as np
from functools import partial


class CouplingLayer(nn.Module):

    def __init__(self, layer_size=2, split_size_x1=1):
        super(CouplingLayer, self).__init__()

        # random split x = (x1, x2)
        choice = np.random.choice(layer_size, split_size_x1, replace=False)
        ind = np.zeros(layer_size, dtype=bool)
        ind[choice] = True
        temp = np.arange(layer_size)
        self.split_index = (temp[ind], temp[~ind])

        self.activation = nn.ReLU()

        # t(x1)
        self.linear1 = nn.Linear(split_size_x1, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, layer_size - split_size_x1)

        # s(x1)
        self.linear5 = nn.Linear(split_size_x1, 100)
        self.linear6 = nn.Linear(100, 100)
        self.linear7 = nn.Linear(100, 100)
        self.linear8 = nn.Linear(100, layer_size - split_size_x1)

    def forward(self, x, reverse=False):
        # x1, x2 = self.split(x)

        t = self.linear1(x[:, self.split_index[0]])
        t = self.activation(t)
        t = self.linear2(t)
        t = torch.sigmoid(t)
        t = self.linear3(t)
        t = self.activation(t)
        t = self.linear4(t)
        t = torch.tanh(t)

        s = self.linear5(x[:, self.split_index[0]])
        s = self.activation(s)
        s = self.linear6(s)
        s = torch.tanh(s)
        s = self.linear7(s)
        s = self.activation(s)
        s = self.linear8(s)
        s = torch.sigmoid(s)

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

    def split(self, x):  # unnecessary
        x1 = x[:, self.split_index[0]]
        x2 = x[:, self.split_index[1]]

        return x1, x2

    def join(self, x1, x2):
        pass


class NN(nn.Module):
    def __init__(self, num_lu_blocks=1, layer_size=2, device="cuda:0"):
        super(NN, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_lu_blocks):
            self.layers.append(CouplingLayer(layer_size, split_size_x1=1))

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
