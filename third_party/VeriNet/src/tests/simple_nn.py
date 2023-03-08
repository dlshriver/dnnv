
"""
A few very simple neural networks used for unit-tests.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import torch.nn as nn

from verinet.neural_networks.verinet_nn import VeriNetNN, VeriNetNNNode
from verinet.neural_networks.custom_layers import Mean, Reshape, MulConstant, AddDynamic


# noinspection PyUnresolvedReferences,PyCallingNonCallable
class SimpleNN(VeriNetNN):

    """
    A simple torch fully connected network.
    """

    def __init__(self, activation="Sigmoid"):

        hidden_1 = nn.Linear(2, 2)
        if activation == "Sigmoid":
            hidden_2 = nn.Sigmoid()
        elif activation == "Tanh":
            hidden_2 = nn.Tanh()
        elif activation == "ReLU":
            hidden_2 = nn.ReLU()
        else:
            raise ValueError(f"Activation function not recognised: {activation}")
        hidden_3 = nn.Linear(2, 1)

        hidden_1.weight.data = torch.Tensor([[1, 2], [-1, 2]])
        hidden_1.bias.data = torch.Tensor([1, 0])

        hidden_3.weight.data = torch.Tensor([[-1, 1]])
        hidden_3.bias.data = torch.Tensor([-1])

        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity(), None, [1]),
                            VeriNetNNNode(1, hidden_1, [0], [2]),
                            VeriNetNNNode(2, hidden_2, [1], [3]),
                            VeriNetNNNode(3, hidden_3, [2], None)]

        super(SimpleNN, self).__init__(verinet_nn_nodes)


# noinspection PyUnresolvedReferences,PyCallingNonCallable
class SimpleNN2Outputs(VeriNetNN):

    """
    A simple torch fully connected network with 2 outputs.
    """

    def __init__(self, activation="Sigmoid"):

        hidden_1 = nn.Linear(2, 2)
        if activation == "Sigmoid":
            hidden_2 = nn.Sigmoid()
        elif activation == "Tanh":
            hidden_2 = nn.Tanh()
        elif activation == "ReLU":
            hidden_2 = nn.ReLU()
        else:
            raise ValueError(f"Activation function not recognised: {activation}")
        hidden_3 = nn.Linear(2, 2)

        hidden_1.weight.data = torch.Tensor([[1, 2], [-1, 2]])
        hidden_1.bias.data = torch.Tensor([1, 0])

        hidden_3.weight.data = torch.Tensor([[1, 0], [0, 1]])
        hidden_3.bias.data = torch.Tensor([0, 0])

        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity(), None, [1]),
                            VeriNetNNNode(1, hidden_1, [0], [2]),
                            VeriNetNNNode(2, hidden_2, [1], [3]),
                            VeriNetNNNode(3, hidden_3, [2], None)]

        super(SimpleNN2Outputs, self).__init__(verinet_nn_nodes)


# noinspection PyUnresolvedReferences,PyCallingNonCallable
class SimpleNN2Layers(VeriNetNN):

    """
    A simple torch fully connected network with 2 outputs.
    """

    def __init__(self, activation="Sigmoid"):

        if activation == "Sigmoid":
            hidden_2 = nn.Sigmoid()
            hidden_4 = nn.Sigmoid()
        elif activation == "Tanh":
            hidden_2 = nn.Tanh()
            hidden_4 = nn.Tanh()
        elif activation == "ReLU":
            hidden_2 = nn.ReLU()
            hidden_4 = nn.ReLU()
        else:
            raise ValueError(f"Activation function not recognised: {activation}")
        hidden_1 = nn.Linear(2, 2)
        hidden_3 = nn.Linear(2, 2)
        hidden_5 = nn.Linear(2, 2)

        hidden_1.weight.data = torch.Tensor([[1, 2], [-1, 2]])
        hidden_1.bias.data = torch.Tensor([1, 0])

        hidden_3.weight.data = torch.Tensor([[1, 0], [0, 1]])
        hidden_3.bias.data = torch.Tensor([0, 0])

        hidden_5.weight.data = torch.Tensor([[1, 0], [0, 1]])
        hidden_5.bias.data = torch.Tensor([0, 0])

        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity(), None, [1]),
                            VeriNetNNNode(1, hidden_1, [0], [2]),
                            VeriNetNNNode(2, hidden_2, [1], [3]),
                            VeriNetNNNode(3, hidden_3, [2], [4]),
                            VeriNetNNNode(4, hidden_4, [3], [5]),
                            VeriNetNNNode(5, hidden_5, [4], None)]

        super(SimpleNN2Layers, self).__init__(verinet_nn_nodes)


# noinspection PyUnresolvedReferences,PyCallingNonCallable,PyTypeChecker
class SimpleNNConv2d(VeriNetNN):

    """
    A simple torch 2-node conv network.
    """

    def __init__(self):

        hidden_1 = nn.Conv2d(1, 2, 3, 1, 1)
        hidden_2 = nn.ReLU()
        hidden_3 = nn.Conv2d(2, 1, 3, 1, 0)

        hidden_1.weight.data = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                                            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        hidden_1.bias.data = torch.Tensor([1, 0])

        hidden_3.weight.data = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        hidden_3.bias.data = torch.Tensor([-1])

        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity(), None, [1]),
                            VeriNetNNNode(1, hidden_1, [0], [2]),
                            VeriNetNNNode(2, hidden_2, [1], [3]),
                            VeriNetNNNode(3, hidden_3, [2], None)]

        super(SimpleNNConv2d, self).__init__(verinet_nn_nodes)


# noinspection PyUnresolvedReferences,PyCallingNonCallable,PyTypeChecker
class SimpleNNBatchNorm2d(VeriNetNN):

    """
    A simple torch 2-node conv network.
    """

    def __init__(self):

        hidden_1 = nn.BatchNorm2d(1)
        hidden_2 = nn.ReLU()
        hidden_3 = nn.Conv2d(1, 1, 3, 1, 0)

        hidden_1.weight.data = torch.Tensor([0.5])
        hidden_1.bias.data = torch.Tensor([0.5])
        hidden_1.running_mean = torch.Tensor([-0.5])
        hidden_1.running_var = torch.Tensor([2])

        hidden_3.weight.data = torch.Tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        hidden_3.bias.data = torch.Tensor([-1])

        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity(), None, [1]),
                            VeriNetNNNode(1, hidden_1, [0], [2]),
                            VeriNetNNNode(2, hidden_2, [1], [3]),
                            VeriNetNNNode(3, hidden_3, [2], None)]

        super(SimpleNNBatchNorm2d, self).__init__(verinet_nn_nodes)


# noinspection PyUnresolvedReferences,PyCallingNonCallable,PyTypeChecker
class SimpleNNAvgPool2d(VeriNetNN):

    """
    A simple torch 2-node conv network.
    """

    def __init__(self):

        hidden_1 = nn.Conv2d(1, 2, 3, 1, 1)
        hidden_2 = nn.ReLU()
        hidden_3 = nn.AvgPool2d(3)

        hidden_1.weight.data = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                                            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        hidden_1.bias.data = torch.Tensor([1, 0])

        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity(), None, [1]),
                            VeriNetNNNode(1, hidden_1, [0], [2]),
                            VeriNetNNNode(2, hidden_2, [1], [3]),
                            VeriNetNNNode(3, hidden_3, [2], None)]

        super(SimpleNNAvgPool2d, self).__init__(verinet_nn_nodes)


# noinspection PyUnresolvedReferences,PyCallingNonCallable,PyTypeChecker
class SimpleNNMean(VeriNetNN):

    """
    A simple torch 2-node conv network.
    """

    def __init__(self):

        hidden_1 = nn.Conv2d(1, 2, 3, 1, 1)
        hidden_2 = nn.ReLU()
        hidden_3 = Mean(dims=(2, 3), keepdim=False)

        hidden_1.weight.data = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
                                            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        hidden_1.bias.data = torch.Tensor([1, 0])

        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity(), None, [1]),
                            VeriNetNNNode(1, hidden_1, [0], [2]),
                            VeriNetNNNode(2, hidden_2, [1], [3]),
                            VeriNetNNNode(3, hidden_3, [2], None)]

        super(SimpleNNMean, self).__init__(verinet_nn_nodes)


# noinspection PyUnresolvedReferences,PyCallingNonCallable,PyTypeChecker
class SimpleNNReshape(VeriNetNN):

    """
    A simple torch 2-node conv network.
    """

    def __init__(self):

        hidden_1 = Reshape((1, 1, 3, 3))
        hidden_2 = nn.ReLU()
        hidden_3 = nn.Conv2d(1, 1, 3, 1, 0)

        hidden_3.weight.data = torch.Tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        hidden_3.bias.data = torch.Tensor([0])

        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity(), None, [1]),
                            VeriNetNNNode(1, hidden_1, [0], [2]),
                            VeriNetNNNode(2, hidden_2, [1], [3]),
                            VeriNetNNNode(3, hidden_3, [2], None)]

        super(SimpleNNReshape, self).__init__(verinet_nn_nodes)


# noinspection PyUnresolvedReferences,PyCallingNonCallable,PyTypeChecker
class SimpleNNMulConstant(VeriNetNN):

    """
    A simple torch 2-node conv network.
    """

    def __init__(self):

        hidden_1 = MulConstant(2)
        hidden_2 = nn.ReLU()
        hidden_3 = nn.Conv2d(1, 1, 3, 1, 0)

        hidden_3.weight.data = torch.Tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        hidden_3.bias.data = torch.Tensor([0])

        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity(), None, [1]),
                            VeriNetNNNode(1, hidden_1, [0], [2]),
                            VeriNetNNNode(2, hidden_2, [1], [3]),
                            VeriNetNNNode(3, hidden_3, [2], None)]

        super(SimpleNNMulConstant, self).__init__(verinet_nn_nodes)


# noinspection PyUnresolvedReferences,PyCallingNonCallable,PyTypeChecker
class SimpleAddDynamic(VeriNetNN):

    """
    A simple torch 2-node conv network.
    """

    def __init__(self):

        hidden_1 = nn.Conv2d(1, 1, 3, 1, 1)
        hidden_2 = nn.Conv2d(1, 1, 3, 1, 1)
        hidden_3 = AddDynamic()
        hidden_4 = nn.Conv2d(1, 1, 3, 1, 0)

        hidden_1.weight.data = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        hidden_1.bias.data = torch.Tensor([0])
        hidden_2.weight.data = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        hidden_2.bias.data = torch.Tensor([0])
        hidden_4.weight.data = torch.Tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        hidden_4.bias.data = torch.Tensor([0])

        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity(), None, [1]),
                            VeriNetNNNode(1, hidden_1, [0], [2, 3]),
                            VeriNetNNNode(2, hidden_2, [1], [3]),
                            VeriNetNNNode(3, hidden_3, [1, 2], [4]),
                            VeriNetNNNode(4, hidden_4, [3], None)]

        super(SimpleAddDynamic, self).__init__(verinet_nn_nodes)


# noinspection PyUnresolvedReferences,PyCallingNonCallable,PyTypeChecker
class SimpleDeepResidual(VeriNetNN):

    """
    A simple torch 2-node conv network.
    """

    def __init__(self):

        hidden_1 = nn.Conv2d(1, 1, 3, 1, 1)
        hidden_2 = nn.ReLU()
        hidden_3 = MulConstant(torch.Tensor([2]))
        hidden_4 = AddDynamic()

        hidden_5 = nn.Conv2d(1, 1, 3, 1, 1)
        hidden_6 = nn.ReLU()
        hidden_7 = MulConstant(torch.Tensor([2]))
        hidden_8 = AddDynamic()

        hidden_9 = nn.Conv2d(1, 1, 3, 1, 1)
        hidden_10 = nn.ReLU()
        hidden_11 = MulConstant(torch.Tensor([2]))
        hidden_12 = AddDynamic()

        hidden_13 = nn.Conv2d(1, 1, 3, 1, 0)

        hidden_1.weight.data = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        hidden_1.bias.data = torch.Tensor([0])

        hidden_5.weight.data = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        hidden_5.bias.data = torch.Tensor([0])

        hidden_9.weight.data = torch.Tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        hidden_9.bias.data = torch.Tensor([0])

        hidden_13.weight.data = torch.Tensor([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]])
        hidden_13.bias.data = torch.Tensor([0])

        verinet_nn_nodes = [VeriNetNNNode(0, nn.Identity(), None, [1, 4]),

                            VeriNetNNNode(1, hidden_1, [0], [2]),
                            VeriNetNNNode(2, hidden_2, [1], [3]),
                            VeriNetNNNode(3, hidden_3, [2], [4]),
                            VeriNetNNNode(4, hidden_4, [0, 3], [5]),

                            VeriNetNNNode(5, hidden_5, [4], [6, 8]),
                            VeriNetNNNode(6, hidden_6, [5], [7]),
                            VeriNetNNNode(7, hidden_7, [6], [8]),
                            VeriNetNNNode(8, hidden_8, [5, 7], [9]),

                            VeriNetNNNode(9, hidden_9, [8], [10, 12]),
                            VeriNetNNNode(10, hidden_10, [9], [11]),
                            VeriNetNNNode(11, hidden_11, [10], [12]),
                            VeriNetNNNode(12, hidden_12, [9, 11], [13]),

                            VeriNetNNNode(13, hidden_13, [12], None)]

        super(SimpleDeepResidual, self).__init__(verinet_nn_nodes)
