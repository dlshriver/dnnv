
"""
Contains custom torch layers.

Author: Patrick Henriksen <patrick@henriksen.as>
"""


import torch
import torch.nn as nn


class Reshape(nn.Module):

    """
    Reshapes the input to the given shape.
    """

    def __init__(self, shape: tuple):
        super(Reshape, self).__init__()
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    def forward(self, x: torch.Tensor):
        return x.reshape(self._shape)


class Mean(nn.Module):

    """
    Reshapes the input to the given shape.
    """

    def __init__(self, dims: tuple, keepdim: bool = False):

        self._dims = dims
        self._keepdim = keepdim

        super(Mean, self).__init__()

    @property
    def dims(self):
        return self._dims

    @property
    def keepdim(self):
        return self._keepdim

    def forward(self, x: torch.Tensor):
        return torch.mean(x, dim=self._dims, keepdim=self._keepdim)


class Crop(nn.Module):
    
    """
    Reshapes the input to the given shape.
    """

    def __init__(self, crop: int = 1):

        self._crop = crop

        super(Crop, self).__init__()

    @property
    def crop(self):
        return self._crop

    def forward(self, x: torch.Tensor):
        return x[..., self.crop:-self.crop, self.crop:-self.crop]


class AddDynamic(nn.Module):

    """
    Adds two tensors from previous layers
    """

    # noinspection PyMethodMayBeStatic
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):

        return x1 + x2


class AddConstant(nn.Module):

    """
    Adds the tensor from a layer with a constant tensor.
    """

    def __init__(self, term: torch.Tensor):

        self.term = term

        super(AddConstant, self).__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x1: torch.Tensor):

        return x1 + self.term


class MulConstant(nn.Module):
    """
    Multiplies the tensor from a layer with a constant tensor.
    """

    def __init__(self, multiplier: torch.Tensor):

        self.multiplier = multiplier

        super(MulConstant, self).__init__()

    def forward(self, x1: torch.Tensor):
        return x1 * self.multiplier


class Transpose(nn.Module):

    """
    Transposes the dimensions of the input tensor
    """

    def __init__(self, dim_order: tuple):

        self.dim_order = dim_order

        super(Transpose, self).__init__()

    def forward(self, x1: torch.Tensor):
        return x1.permute(self.dim_order)


class Unsqueeze(nn.Module):

    """
    Transposes the dimensions of the input tensor
    """

    def __init__(self, dims: tuple):

        self.dims = dims

        super(Unsqueeze, self).__init__()

    # noinspection PyTypeChecker
    def forward(self, x1: torch.Tensor):

        for dim in sorted(self.dims):
            x1 = x1.unsqueeze(dim)

        return x1

    def new_shape(self, old_shape: tuple) -> list:

        new_shape = [1] * (len(self.dims) + len(old_shape))

        j = 0
        for i in range(len(new_shape)):

            if i in self.dims:
                continue
            else:
                new_shape[i] = old_shape[j]
                j += 1

        return new_shape
