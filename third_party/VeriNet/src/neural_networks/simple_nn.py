
"""
A few very simple neural networks used for unit-tests.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import torch.nn as nn

from src.neural_networks.verinet_nn import VeriNetNN


class SimpleNN(VeriNetNN):

    """
    A simple torch fully connected neural network with 2 input nodes, 2 hidden and 1 output for testing.
    """

    def __init__(self, activation="Sigmoid"):

        """
        Args:
            activation: The activation function, choose from ["Sigmoid", "Tanh", "Relu"]
        """

        if activation == "Sigmoid":
            hidden_1 = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())
            hidden_2 = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        elif activation == "Tanh":
            hidden_1 = nn.Sequential(nn.Linear(2, 2), nn.Tanh())
            hidden_2 = nn.Sequential(nn.Linear(2, 1), nn.Tanh())
        else:
            hidden_1 = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
            hidden_2 = nn.Sequential(nn.Linear(2, 1), nn.ReLU())

        # noinspection PyArgumentList
        list(hidden_1.children())[0].weight.data = torch.Tensor([[1, 2], [-1, 2]])
        # noinspection PyArgumentList
        list(hidden_1.children())[0].bias.data = torch.Tensor([1, 0])
        # noinspection PyArgumentList
        list(hidden_2.children())[0].weight.data = torch.Tensor([[-1, 1]])
        # noinspection PyArgumentList
        list(hidden_2.children())[0].bias.data = torch.Tensor([-1])

        self.layer_sizes = [2, 2, 1]
        self.layer_activations = ["Linear"] + [activation] * 2

        super(SimpleNN, self).__init__([hidden_1, hidden_2])


class SimpleNN2(VeriNetNN):

    """
    A simple torch fully connected neural network with 1 input, 2 hidden and 2 output and 2 output nodes.
    Mainly used for testing adversarial verifications in VeriNet.
    """

    def __init__(self, activation: str="Sigmoid"):

        if activation == "Sigmoid":
            hidden_1 = nn.Sequential(nn.Linear(1, 2), nn.Sigmoid())
            hidden_2 = nn.Sequential(nn.Linear(2, 2))
        elif activation == "Tanh":
            hidden_1 = nn.Sequential(nn.Linear(1, 2), nn.Tanh())
            hidden_2 = nn.Sequential(nn.Linear(2, 2))
        else:
            hidden_1 = nn.Sequential(nn.Linear(1, 2), nn.ReLU())
            hidden_2 = nn.Sequential(nn.Linear(2, 2))

        # noinspection PyArgumentList

        list(hidden_1.children())[0].weight.data = torch.Tensor([[1], [-1]])
        # noinspection PyArgumentList
        list(hidden_1.children())[0].bias.data = torch.Tensor([0.5, 0.5])
        # noinspection PyArgumentList
        list(hidden_2.children())[0].weight.data = torch.Tensor([[1, -1], [-1, 1]])
        # noinspection PyArgumentList
        list(hidden_2.children())[0].bias.data = torch.Tensor([0.5, 0])

        super().__init__([hidden_1, hidden_2])


class SimpleNNConv2(VeriNetNN):

    """
    A simple torch fully connected neural network with 1 input, 2 hidden and 2 output and 2 output nodes.
    Mainly used for testing adversarial verifications in VeriNet.
    """

    # noinspection PyArgumentList
    def __init__(self):

        hidden_1 = nn.Sequential(nn.Conv2d(1, 2, 3, 2, 1))
        hidden_2 = nn.Sequential(nn.Conv2d(2, 1, 3, 1, 0))

        list(hidden_1.children())[0].weight.data = torch.ones((2, 1, 3, 3))
        list(hidden_1.children())[0].weight.data[1, :, :] += 1
        list(hidden_2.children())[0].weight.data = torch.ones((1, 2, 3, 3))

        list(hidden_1.children())[0].bias.data = torch.Tensor([10, 20])
        list(hidden_2.children())[0].bias.data = torch.Tensor([30])

        super().__init__([hidden_1, hidden_2])


class SimpleNNBatchNorm2D(VeriNetNN):

    """
    A simple torch network with batch norm used for testing
    """

    # noinspection PyArgumentList
    def __init__(self):

        hidden_1 = nn.Sequential(nn.Conv2d(1, 2, 1, 1, 0))
        hidden_2 = nn.Sequential(nn.BatchNorm2d(2))

        list(hidden_1.children())[0].weight.data = torch.ones((2, 1, 1, 1))
        list(hidden_1.children())[0].bias.data = torch.Tensor([0, 0])

        list(hidden_2.children())[0].weight.data = torch.Tensor([0.5, 2])
        list(hidden_2.children())[0].bias.data = torch.Tensor([1, 2])
        list(hidden_2.children())[0].running_mean = torch.Tensor([0.2, 1])
        list(hidden_2.children())[0].running_var = torch.Tensor(([3, 5]))

        super().__init__([hidden_1, hidden_2])
