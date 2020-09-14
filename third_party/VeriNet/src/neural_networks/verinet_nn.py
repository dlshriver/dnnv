
"""
Contains the neural networks used in VeriNet.

Author: Patrick Henriksen <patrick@henriksen.as>
"""


from typing import Callable

import torch
import torch.nn as nn


class VeriNetNN(nn.Module):

    """
    The torch model used in VeriNet.

    The preferred way of creating a neural network for VeriNet is to directly use this class by passing a list
    with the layers/ activation functions to init.

    Subclassing is also possible; however, the self.layers, self.logits and the forward function are essential
    for the verification process and should not be altered.
    """

    def __init__(self, layers: list, out_activation: Callable=None, use_gpu: bool=False):

        """
        Args:
            layers          : A list with the layers/ activation functions of the model.
            out_activation  : The activation function applied to the output. Any activation function can be used,
                              however all verification is done on the logits, the values before this function is
                              applied
            use_gpu         : If true, and a GPU is available, the GPU is used, else the CPU is used

        """

        super().__init__()
        self.layers = nn.ModuleList(self._flatten_layers(layers))
        self.out_activation = out_activation
        self.logits = None
        self.device = None

        self._set_device(use_gpu=use_gpu)

    def _flatten_layers(self, layers_in: list) -> list:

        """
        Removes all sequential objects from self.layers and returns the content as a list.

        Args:
            layers_in: The torch.nn layer/activation function
        Returns:
            A list with all objects in layers_in, where nn.Sequential objects are removed and the contents are added
            to the list instead.
        """

        layers_out = []

        for layer in layers_in:
            layers_out += self._flatten_layers_rec(layer)

        return layers_out

    def _flatten_layers_rec(self, layer) -> list:

        """
        Recursively iterating through nn.Sequential objects and returning all content as a list.

        Args:
            layer: The torch.nn layer/activation function
        Returns:
            If layer is not a nn.Sequential, [layer] is returned. Otherwise the content of the Sequential object is
            recursively extracted and a list with all the content is returned.
        """

        layers = []

        if isinstance(layer, nn.Sequential):
            for child in layer.children():
                layers += self._flatten_layers_rec(child)
        else:
            layers.append(layer)

        return layers

    def _set_device(self, use_gpu: bool):

        """
        Initializes the gpu/ cpu

        Args:
            use_gpu: If true, and a GPU is available, the GPU is used, else the CPU is used
        """

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward calculations for the network.

        Each object in the self.layers list is applied sequentially.

        Args:
            x   : The input, should be BxN for FC or BxNxHxW for Conv2d, where B is the batch size, N is the number
                  of nodes, H his the height and W is the width.

        Returns:
            The network output, of same shape as the input
        """

        if len(x.shape) == 1:

            # Add batch dimension for FC input
            batch_size = 1
            x = x.view(1, -1)

        elif len(x.shape) == 3:

            # Add batch dimension for conv2d input
            batch_size = 1
            x = x.view(1, *x.shape)

        else:
            batch_size = x.shape[0]

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = x.reshape(batch_size, -1)
            x = layer(x)

        x = x.reshape(batch_size, -1)

        self.logits = x.clone()

        if self.out_activation is not None:
            return self.out_activation(x)
        else:
            return x

    def save(self, path: str):

        """
        Saves the network to the given path in torch.save format.

        Args:
            path: The path
        """

        torch.save(self.state_dict(), path)

    # noinspection PyUnresolvedReferences
    def load(self, path: str):

        """
        Loads a network network in torch.save format from the given path

        Args:
             path: The path
        """

        self.load_state_dict(torch.load(path, map_location='cpu'))
