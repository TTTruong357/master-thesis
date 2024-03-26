import torch
from tqdm import tqdm

from functions import gaussian_loss_function, uniform_circle_log_likelihood_method2
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path
from model import register_activation_hooks, remove_activation_hooks

"""testing with gaussian data"""


def testing_routine(model, test_loader, loss_function=gaussian_loss_function):
    model.eval()
    test_loss = 0

    """outputs of L-layer are needed for the loss function"""
    layers = []
    for j in range(len(model.intermediate_lu_blocks)):
        if j % 2 != 0:
            layers.append("intermediate_lu_blocks.{}".format(j))
    layers.append("final_lu_block.1")

    with torch.no_grad():
        for i, input in tqdm(enumerate(test_loader)):
            saved_layers, handles = register_activation_hooks(model, layers_to_save=layers)
            output = model(input)
            loss = loss_function(output, model, saved_layers)
            test_loss += loss
            remove_activation_hooks(handles)

    print('Test set: Average loss: {:.4f}'.format(test_loss / len(test_loader)))


def plot_distribution(output, save_name):
    output_data = output.detach().flatten().cpu().numpy()
    """remove outlier"""
    upper = np.quantile(output_data, 0.997)
    lower = np.quantile(output_data, 0.003)
    output_data = output_data[np.logical_and(output_data > lower, output_data < upper)]

    plt.hist(output_data, density=True, label="forward distribution")

    x = np.linspace(-3, 3, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), label="pdf standard normal")
    plt.legend()

    plt.xlim(-3.3, 3.3)
    plt.ylim(0, 0.6)
    save_path = Path("outputs/plots/" + save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()
