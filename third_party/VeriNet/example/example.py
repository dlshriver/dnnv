import numpy as np
import torch
import gurobipy as grb
import torch.nn as nn

from src.neural_networks.verinet_nn import VeriNetNN
from src.algorithm.verinet import VeriNet
from src.data_loader.input_data_loader import load_neurify_mnist
from src.data_loader.nnet import NNET
from src.algorithm.verification_objectives import LocalRobustnessObjective
from src.algorithm.verinet_util import Status


def create_input_bounds(image: np.array, eps: int):

    """
    Creates the l-infinity input bounds from the given image and epsilon
    """

    input_bounds = np.zeros((*image.shape, 2), dtype=np.float32)
    input_bounds[:, 0] = image - eps
    input_bounds[:, 1] = image + eps

    return input_bounds


def local_robustnes_nnet():

    """
    An example run of the local robustness verification objective using nnet.
    """

    print("\nRunning example run for local robustness verification objective:")

    # Load the nnet and convert to VeriNetNN (Found in src/neural_networks/verinet_nn.py
    nnet = NNET("../data/models_nnet/neurify/mnist24.nnet")
    model = nnet.from_nnet_to_verinet_nn()

    # Initialize the solver
    solver = VeriNet(model, max_procs=20)

    # Load the image and use the predicted class as correct class
    image = load_neurify_mnist("../data/mnist_neurify/test_images_100/", img_nums=[0]).reshape(-1)
    correct_class = int(model(torch.Tensor(image)).argmax(dim=1))

    for eps in [8, 15]:

        # Create the input bounds
        input_bounds = create_input_bounds(image, eps)
        input_bounds = nnet.normalize_input(input_bounds)

        # Initialize the verification objective and solve the problem
        objective = LocalRobustnessObjective(correct_class, input_bounds, output_size=10)
        solver.verify(objective, timeout=3600, no_split=False, verbose=False)

        # Store the counter example if Unsafe. Status enum is defined in src.algorithm.verinet_util
        if solver.status == Status.Unsafe:
            _ = solver.counter_example

        print("")
        print(f"Statistics for epsilon = {eps}:")
        print(f"Verification results: {solver.status}")
        print(f"Branches explored: {solver.branches_explored}")
        print(f"Maximum depth reached: {solver.max_depth}")


def verinet_nn_example():

    """
    An example run of how to use the VeriNetNN class to create a neural network instead of reading from nnet file.

    The VeriNetNN class accepts a list of layers arg in init. The forward function should not be modified and it
    is assumed that each object in the layers list is applied sequentially.
    """

    print("\nRunning example run with custom VeriNetNN and the local robustness verification objective:")

    torch.manual_seed(0)
    layers = [nn.Linear(2, 2),
              nn.ReLU(),
              nn.Linear(2, 2),
              nn.ReLU(),
              nn.Linear(2, 2)]

    model = VeriNetNN(layers)

    input_bounds = np.array([[-10, 10],
                            [-10, 10]])

    solver = VeriNet(model, max_procs=20)
    objective = LocalRobustnessObjective(correct_class=1, input_bounds=input_bounds, output_size=2)
    solver.verify(objective, timeout=3600, no_split=False, verbose=False)

    # Store the counter example if Unsafe. Status enum is defined in src.algorithm.verinet_util
    if solver.status == Status.Unsafe:
        _ = solver.counter_example

    print("")
    print(f"Verification results: {solver.status}")
    print(f"Branches explored: {solver.branches_explored}")
    print(f"Maximum depth reached: {solver.max_depth}")


if __name__ == '__main__':

    # Get the "Academic license" print from gurobi at the beginning
    grb.Model()

    local_robustnes_nnet()
    verinet_nn_example()
