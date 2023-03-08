
"""
Contains the neural networks used in VeriNet.

Author: Patrick Henriksen <patrick@henriksen.as>
"""


from copy import deepcopy

import torch
import torch.nn as nn
import torch.onnx as tonnx


# noinspection PyTypeChecker
class VeriNetNN(nn.Module):

    """
    The torch _model used in VeriNet.

    The preferred way of creating a neural network for VeriNet is to save the _model
    in onnx format and use the load() function of this class.
    """

    def __init__(self, nodes: list, use_gpu: bool = False):

        """
        Args:
            nodes:
                A list of VeriNetNNNodes
            use_gpu:
                If true, and a GPU is available, the GPU is used, else the CPU is used
        """

        super().__init__()

        self.nodes = nodes

        layers = []
        for node in nodes:
            layers.append(node.op)

        self.layers = nn.ModuleList(layers)
        self.uses_gpu = None
        self.set_device(use_gpu=use_gpu)

    @property
    def uses_64bit(self):

        """
        Returns true if at least one parameter uses double instead of float.
        """

        for param in self.parameters():
            if isinstance(param, torch.DoubleTensor):
                return True

        return False

    # noinspection PyAttributeOutsideInit
    def set_device(self, use_gpu: bool):

        """
        Initializes the gpu/ cpu

        Args:
            use_gpu:
                If true, and a GPU is available, the GPU is used, else the CPU is used
        """

        self.uses_gpu = use_gpu

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.to(device=self.device)

    def forward(self, x: torch.Tensor, cleanup: bool = True) -> torch.Tensor:

        """
        Forward calculations for the network.

        Args:
            x:
                The input, should be BxN for FC or BxNxHxW for Conv2d, where B is the
                batch size, N is the number of nodes, H his the height and W is the
                width.
            cleanup:
                If true, intermediate results are deleted when not needed anymore.

        Returns:
            The network output, of same shape as the input.
        """

        if len(x.shape) != 2 and len(x.shape) != 4:
            raise ValueError("Input should have 2 or 4 dimensions where the first dimension is the batch")

        self.nodes[0].value = x

        for node_num, node in enumerate(self.nodes[1:]):
            node.input_value = [self.nodes[i].value.clone() for i in node.connections_from]
            node.value = node(node.input_value)

            if cleanup:
                self.cleanup(node_num)

        return [node.value for node in self.nodes if len(node.connections_to) == 0]

    def cleanup(self, node_num: int):

        """
        Removes computed values from all nodes that do not have a connection
        past node_num.

        Args:
            node_num:
                The number of the current node.
        """

        for node in self.nodes:

            node.input_value = None

            if len(node.connections_to) == 0:  # Output node
                continue

            remaining_connects = [i for i in node.connections_to if i > node_num]

            if len(remaining_connects) == 0:
                node.value = None

    # noinspection PyTypeChecker
    def save(self, dummy_in: torch.Tensor, path: str = "../../resources/networks/"):

        """
        Saves the _model to onnx format.

        Args:
            dummy_in:
                A dummy tensor of the same shape as the real inputs. For example,
                if the input has shape (3, 32, 32) use:

                torch.randn(1, 3, 32, 32).to(device=self.device)

            path:
                The save path.
        """

        tonnx.export(self, dummy_in, path, verbose=False, opset_version=9)

    @staticmethod
    def load_onnx(path: str):

        """
        Loads an onnx _model.

        Args:
            path: The path of the onnx _model.
        Returns:
            The VeriNetNN object
        """

        from verinet.parsers.onnx_parser import ONNXParser

        parser = ONNXParser(path)
        return parser.to_pytorch()

    def save_sd(self, path: str):

        """
        Saves the models state dict.

        Args:
            path:
                The save path.
        """

        torch.save(self.state_dict(), path)

    def load_sd(self, path: str):

        """
        Loads the models state dict.

        Args:
            path:
                The save path.
        """

        self.load_state_dict(torch.load(path))


class VeriNetNNNode:

    """
    This class stores a pytorch operation as well as the connected nodes and computed
    values.
    """

    def __init__(self, idx: int, op: torch.nn, connections_from: list = None, connections_to: list = None):

        """
        Args:
            idx:
                The index of the node
            op:
                The torch.nn operation.
            connections_from:
                The indices of nodes that are connected to this node.
            connections_to:
                The indices of nodes this node is connected to.
        """

        self.idx = idx
        self.op = op

        self.connections_to = connections_to if connections_to is not None else []
        self.connections_from = connections_from if connections_from is not None else []

        self.value = None
        self.input_value = None

    def __call__(self, x: torch.Tensor):
        return self.op(*x)

    def __str__(self):
        return f"VeriNetNNNode(idx: {self.idx}, op: {self.op}, " \
               f"to: {self.connections_to}, from: {self.connections_from})"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return VeriNetNNNode(self.idx, deepcopy(self.op), self.connections_from.copy(), self.connections_to.copy())

    def copy(self):
        return self.__copy__()
