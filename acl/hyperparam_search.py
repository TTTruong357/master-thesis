from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, trials_from_docs


from torch.optim.lr_scheduler import StepLR
from train import training_routine
from functions import uniform_circle_loss_function

import torch


def objective(space):
    """
    Hier die Funktion einfügen, wo die Hyperparamter optimiert werden sollen

    Input: space, type: dictionary mit den Hyperparametern
    Output: Der Score/ die Accuracy
    """

    # collect all parameters
    NUM_COUPLING_LAYERS = space['num_couling_layers']

    LEARNING_RATE = space['learning_rate']
    MOMENTUM = space['momentum']
    STEP_SIZE = space['step_size']
    GAMMA = space['gamma']

    NUM_EPOCH = space['num_epoch']
    BATCH_SIZE = 2 ** space['batch_size_exponent']

    NN_SIZE = (space['NN_depth'], space['NN_width'])

    # DENSITY_PARAM_INCREMENT = space['density_param_increment']

    print(space['batch_size_exponent'])

    # make calculations

    torch.manual_seed(0)
    model = NN(num_coupling_layers=NUM_COUPLING_LAYERS, layer_size=2, split_size_x1=1, nn_size=NN_SIZE).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    num_epoch = NUM_EPOCH
    batch_size = BATCH_SIZE
    for epoch in range(num_epoch):
        # density_param = (1+DENSITY_PARAM_INCREMENT*epoch, 0.2)
        density_param = (1 + 10 * (epoch + 1) / num_epoch, 0.2)
        # print(density_param, epoch, num_epoch)
        training_routine_uniform(model, device, train_loader, optimizer, epoch, batch_size,
                                 uniform_circle_loss_function, density_param)

    density_param = (11., 0.2)
    test_score = testing_routine_uniform(model, device, test_loader, batch_size, uniform_circle_loss_function,
                                         density_param)

    #
    # accuracy = test_score #calculate some kind of score/ accuracy

    # #normalisiere am besten die Accuracy! (so dass es im (0,1) intervall ist)
    # accuracy = accuracy

    # We aim to maximize accuracy, therefore we return it as a negative value
    # Wenn also etwas maximiert werden soll, schreibe ein - im loss. Sonst lasse das minus weg
    # Man kann sich hier zusätzlich in der action_info Sachen merken.
    return {'loss': test_score, 'status': STATUS_OK}


def objective_method2(space):
    """
    Hier die Funktion einfügen, wo die Hyperparamter optimiert werden sollen

    Input: space, type: dictionary mit den Hyperparametern
    Output: Der Score/ die Accuracy
    """

    # collect all parameters
    NUM_COUPLING_LAYERS = space['num_couling_layers']

    LEARNING_RATE = space['learning_rate']
    MOMENTUM = space['momentum']
    STEP_SIZE = space['step_size']
    GAMMA = space['gamma']

    NUM_EPOCH = space['num_epoch']
    BATCH_SIZE = 2 ** space['batch_size_exponent']

    NN_SIZE = (space['NN_depth'], space['NN_width'])

    print(space['batch_size_exponent'])

    # make calculations

    torch.manual_seed(0)
    model = NN(num_coupling_layers=NUM_COUPLING_LAYERS, layer_size=2, split_size_x1=1, nn_size=NN_SIZE).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    num_epoch = NUM_EPOCH
    batch_size = BATCH_SIZE
    for epoch in range(num_epoch):
        training_routine(model, device, train_loader, optimizer, epoch, batch_size,
                         uniform_circle_loss_function_method2)

    test_score = testing_routine(model, device, test_loader, batch_size, uniform_circle_loss_function_method2)

    #
    # accuracy = test_score #calculate some kind of score/ accuracy

    # #normalisiere am besten die Accuracy! (so dass es im (0,1) intervall ist)
    # accuracy = accuracy

    # We aim to maximize accuracy, therefore we return it as a negative value
    # Wenn also etwas maximiert werden soll, schreibe ein - im loss. Sonst lasse das minus weg
    # Man kann sich hier zusätzlich in der action_info Sachen merken.
    return {'loss': test_score, 'status': STATUS_OK}