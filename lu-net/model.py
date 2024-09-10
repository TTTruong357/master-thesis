import torch
import torch.nn as nn
from functions import LeakySoftplus, InvertedLeakySoftplus
from functools import partial
from pathlib import Path

def get_zero_grad_hook(mask, device="cuda:0"):
    """zero out gradients"""

    def hook(grad):
        return grad * mask.to(device)

    return hook


class LUNet(nn.Module):
    def __init__(self, num_lu_blocks=1, layer_size=2, device="cuda:0"):
        super(LUNet, self).__init__()

        """masks to zero out gradients"""
        self.mask_triu = torch.triu(torch.ones(layer_size, layer_size)).bool()
        self.mask_tril = torch.tril(torch.ones(layer_size, layer_size)).bool().fill_diagonal_(False)
        # print(self.mask_triu, self.mask_tril)
        self.nonlinearity = LeakySoftplus()
        self.inverted_nonlinearity = InvertedLeakySoftplus

        mask = torch.diag(torch.ones(layer_size))

        """create LU modules"""
        self.intermediate_lu_blocks = nn.ModuleList()
        """adding number of LU Blocks"""
        for i in range(num_lu_blocks):
            """init upper triangular weight matrix U without bias"""
            self.intermediate_lu_blocks.append(nn.Linear(layer_size, layer_size, bias=False))
            upper = self.intermediate_lu_blocks[-1]
            with torch.no_grad():
                upper.weight.copy_(torch.triu(upper.weight))  # set lower weights to zero
                upper.weight = torch.nn.Parameter(
                    mask * torch.diag(torch.rand(layer_size) + 0.1) + (1. - mask) * upper.weight)
            upper.weight.register_hook(
                get_zero_grad_hook(self.mask_triu, device))  # tell pytorch to not use lower weights in training
            """init lower triangular weight matrix L with bias"""
            self.intermediate_lu_blocks.append(nn.Linear(layer_size, layer_size))
            lower = self.intermediate_lu_blocks[-1]
            with torch.no_grad():
                lower.weight.copy_(torch.tril(lower.weight))
                lower.weight.copy_(lower.weight.fill_diagonal_(1))
            lower.weight.register_hook(get_zero_grad_hook(self.mask_tril, device))

        """Adding one final LU block = extra block"""
        self.final_lu_block = nn.ModuleList()
        """init upper triangular weight matrix U without bias"""
        self.final_lu_block.append(nn.Linear(layer_size, layer_size, bias=False))
        upper = self.final_lu_block[-1]
        with torch.no_grad():
            upper.weight.copy_(torch.triu(upper.weight))
            upper.weight = torch.nn.Parameter(
                mask * torch.diag(torch.rand(layer_size) + 0.1) + (1. - mask) * upper.weight)
        upper.weight.register_hook(get_zero_grad_hook(self.mask_triu, device))
        """init lower triangular weight matrix L with bias"""
        self.final_lu_block.append(nn.Linear(layer_size, layer_size))
        lower = self.final_lu_block[-1]
        with torch.no_grad():
            lower.weight.copy_(torch.tril(lower.weight))
            lower.weight.copy_(lower.weight.fill_diagonal_(1))
        lower.weight.register_hook(get_zero_grad_hook(self.mask_tril, device))

    def forward(self, x, reverse=False):
        if not reverse:
            for i, layer in enumerate(self.intermediate_lu_blocks):
                x = layer(x)
                if i % 2 == 1:
                    x = self.nonlinearity(x)

            """final LU block without activation"""
            for i, layer in enumerate(self.final_lu_block):
                x = layer(x)
            return x
        else:
            """final LU-block without activation"""
            for i, layer in reversed(list(enumerate(self.final_lu_block))):
                if i % 2 == 1:
                    x = x - layer.bias
                    x = torch.linalg.solve_triangular(layer.weight, x.T, upper=False)
                if i % 2 == 0:
                    x = torch.linalg.solve_triangular(layer.weight, x, upper=True)

            """all intermediate LU-blocks in reversed order"""
            l = []
            for i, layer in reversed(list(enumerate(self.intermediate_lu_blocks))):
                if i % 2 == 1:
                    if len(l) == 0:
                        l.append(self.inverted_nonlinearity.apply(x))
                    else:
                        l.append(self.inverted_nonlinearity.apply(l[-1]))

                    l[-1] = l[-1] - layer.bias.reshape(2, 1)
                    l[-1] = torch.linalg.solve_triangular(layer.weight, l[-1], upper=False)

                if i % 2 == 0:
                    l[-1] = torch.linalg.solve_triangular(layer.weight, l[-1], upper=True)

            return l[-1].T


"""
* helper functions to store activations and parameters in intermediate layers of the model
* use forward hooks for this, which are functions executed automatically during forward pass
* in PyTorch hooks are registered for nn.Module and are triggered by forward pass of object
"""


def save_activations(activations_dict, name, module, input, output):
    activations_dict[name].append(output)


def register_activation_hooks(model, layers_to_save):
    """Register forward hooks in specified layers"""
    activations_dict = {name: [] for name in layers_to_save}
    handles = []
    for name, module in model.named_modules():
        if name in layers_to_save:
            handle = module.register_forward_hook(partial(save_activations, activations_dict, name))
            handles.append(handle)
    return activations_dict, handles


def remove_activation_hooks(handles):
    """Remove registered hooks"""
    for handle in handles:
        handle.remove()


"""
save and load the model
"""


def save_model(model, data='Ellipse', checkpoint_number=31):
    checkpoints_dir = './lu_uniform/'
    save_path = Path(checkpoints_dir) / Path("{}/experiment{}.pth".format(data, checkpoint_number))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Saved checkpoint:", save_path)


def load_model(device, num_lu_blocks=12, layer_size=2, path=f"./lu_uniform/Ellipse/experiment{1}.pth"):
    model = LUNet(num_lu_blocks=num_lu_blocks, layer_size=layer_size).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
