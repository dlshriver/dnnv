"""
This file contains abstractions for layers (FC, Conv ...).

The abstractions are used to calculate linear relaxations, function values, and
derivatives.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from verinet.sip_torch.operations.abstract_operation import AbstractOperation
from verinet.neural_networks.custom_layers import Reshape as ReshapeTorch
from verinet.neural_networks.custom_layers import Mean as MeanTorch
from verinet.neural_networks.custom_layers import Crop as CropTorch
from verinet.neural_networks.custom_layers import MulConstant as MulConstantTorch
from verinet.neural_networks.custom_layers import AddDynamic as AddDynamicTorch
from verinet.neural_networks.custom_layers import Transpose as TransposeTorch
from verinet.neural_networks.custom_layers import AddConstant as AddConstantTorch
from verinet.neural_networks.custom_layers import Unsqueeze as UnsqueezeTorch


class Identity(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return True

    @property
    def required_params(self) -> list:
        return []

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [nn.Identity]

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        return x

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        return bounds_symbolic_pre[0]

    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        return [bounds_symbolic_post]

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)


class Reshape(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return True

    @property
    def required_params(self) -> list:
        return ["shape"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [ReshapeTorch]

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        return x

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        return bounds_symbolic_pre[0]

    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        return [bounds_symbolic_post]

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)

    # noinspection PyArgumentList,PyCallingNonCallable
    def out_shape(self, in_shape: torch.Tensor) -> torch.Tensor:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        return torch.LongTensor([int(x) for x in self.params["shape"][1:]])


class Transpose(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return True

    @property
    def required_params(self) -> list:
        return ["dim_order"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [TransposeTorch]

    # noinspection PyTypeChecker
    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a list of NxM tensors where each row is a symbolic bounds
                for the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        if len(x) != 1:
            raise ValueError("Expected one set of bounds for transpose")

        dim_order = [dim - 1 for dim in self.params["dim_order"] if dim != 0]  # Remove batchdim

        x = x[0]
        x = x.reshape((*self.params["in_shape"], x.shape[-1]))
        x = x.permute((*dim_order, len(x.shape) - 1))
        x = x.reshape((-1, x.shape[-1]))

        return x

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        if len(bounds_symbolic_pre) != 1:
            raise ValueError("Expected one set of bounds for transpose")

        dim_order = [dim - 1 for dim in self.params["dim_order"] if dim != 0]  # Remove batchdim

        bounds_new = torch.zeros_like(bounds_symbolic_pre[0])
        bounds = bounds_symbolic_pre[0].reshape((2, *self.params["in_shape"], bounds_symbolic_pre[0].shape[-1]))
        bounds_new[0, :] = bounds[0, :].permute((*dim_order,
                                                 len(bounds[0].shape) - 1)).reshape((1, -1, bounds.shape[-1]))
        bounds_new[1, :] = bounds[1, :].permute((*dim_order,
                                                 len(bounds[1].shape) - 1)).reshape((1, -1, bounds.shape[-1]))

        return bounds_new

    # noinspection PyCallingNonCallable
    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        coeffs = bounds_symbolic_post[:, :-1]

        out_shape = self.out_shape(self.params["in_shape"])
        dim_order = [dim for dim in self.params["dim_order"] if dim != 0]  # Remove batchdim
        inv_dim_order = tuple(int(x) + 1 for x in torch.argsort(torch.LongTensor(dim_order)))

        coeffs = coeffs.reshape((coeffs.shape[0], *out_shape))
        coeffs = coeffs.permute((0, *inv_dim_order))
        coeffs = coeffs.reshape((bounds_symbolic_post.shape[0], -1))

        return [torch.cat((coeffs, bounds_symbolic_post[:, -1].unsqueeze(1)), dim=1)]

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)

    # noinspection PyArgumentList,PyCallingNonCallable
    def out_shape(self, in_shape: torch.Tensor) -> torch.Tensor:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        dim_order = [dim - 1 for dim in self.params["dim_order"] if dim != 0]  # Remove batchdim

        return torch.LongTensor(in_shape[list(dim_order)])


class Flatten(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return True

    @property
    def required_params(self) -> list:
        return []

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [nn.Flatten]

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        return x

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        return bounds_symbolic_pre[0]

    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        return [bounds_symbolic_post]

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)

    # noinspection PyArgumentList,PyCallingNonCallable
    def out_shape(self, in_shape: torch.Tensor) -> torch.Tensor:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        return torch.LongTensor((int(torch.prod(in_shape)), ))


class FC(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return False

    @property
    def required_params(self) -> list:
        return ["weight", "bias"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [nn.modules.linear.Linear]

    def forward(self, x: torch.Tensor, add_bias: bool = True, calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The value of the activation function at const_terms.
        """

        if calc_nodes is None:
            y = self.params["weight"] @ x

            if add_bias and self.params["bias"] is not None:
                y[:, -1] += self.params["bias"]
        else:
            y = torch.zeros((self.params["weight"].shape[0], x.shape[1]), dtype=self._precision).to(device=x.device)
            y[calc_nodes] = self.params["weight"][calc_nodes] @ x

            if add_bias and self.params["bias"] is not None:
                y[:, -1][calc_nodes] += self.params["bias"][calc_nodes]

        return y

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        if len(bounds_symbolic_pre) > 1:
            raise ValueError(f"Expected one set of symbolic bounds, got {len(bounds_symbolic_pre)}")
        bounds_symbolic_pre = bounds_symbolic_pre[0]

        bounds_symbolic_post = torch.zeros((2, self.params["weight"].shape[0], bounds_symbolic_pre.shape[2]),
                                           dtype=self._precision).to(device=bounds_symbolic_pre.device)

        pos_weigths, neg_weights = self.params["weight"].clone(), self.params["weight"].clone()
        pos_weigths[pos_weigths < 0] = 0
        neg_weights[neg_weights > 0] = 0

        if calc_nodes is None:

            bounds_symbolic_post[0] = pos_weigths @ bounds_symbolic_pre[0]
            bounds_symbolic_post[0] += neg_weights @ bounds_symbolic_pre[1]
            bounds_symbolic_post[1] = pos_weigths @ bounds_symbolic_pre[1]
            bounds_symbolic_post[1] += neg_weights @ bounds_symbolic_pre[0]

            if add_bias:
                bounds_symbolic_post[:, :, -1] += self.params["bias"]
        else:

            bounds_symbolic_post[0][calc_nodes] = pos_weigths[calc_nodes] @ bounds_symbolic_pre[0]
            bounds_symbolic_post[0][calc_nodes] += neg_weights[calc_nodes] @ bounds_symbolic_pre[1]
            bounds_symbolic_post[1][calc_nodes] = pos_weigths[calc_nodes] @ bounds_symbolic_pre[1]
            bounds_symbolic_post[1][calc_nodes] += pos_weigths[calc_nodes] @ bounds_symbolic_pre[0]

            if add_bias:
                bounds_symbolic_post[:, :, -1][calc_nodes] += self.params["bias"][calc_nodes]

        return bounds_symbolic_post

    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        weights = self.params["weight"]
        bias = self.params["bias"]

        bounds_symbolic_pre = torch.empty((bounds_symbolic_post.shape[0], weights.shape[1] + 1),
                                          dtype=self._precision).to(device=bounds_symbolic_post.device)
        bounds_symbolic_pre[:, -1] = bounds_symbolic_post[:, -1]
        bounds_symbolic_pre[:, :-1] = torch.einsum('ki, ij->kj', bounds_symbolic_post[:, :-1], weights)

        if add_bias:
            bounds_symbolic_pre[:, -1] += torch.sum(bias.view(1, -1) * bounds_symbolic_post[:, :-1], dim=1)

        return [bounds_symbolic_pre]

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)

    # noinspection PyArgumentList,PyCallingNonCallable
    def out_shape(self, in_shape: torch.Tensor) -> torch.Tensor:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        return torch.LongTensor((self.params["weight"].shape[0], ))


class Conv2d(AbstractOperation):

    def __init__(self):

        self._out_unfolded = None
        super().__init__()

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return False

    @property
    def required_params(self) -> list:
        return ["weight", "bias", "kernel_size", "padding", "stride", "in_channels", "out_channels", "groups"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [nn.Conv2d]

    # noinspection PyArgumentList
    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        stride = self.params["stride"]
        padding = self.params["padding"]
        weights = self.params["weight"]
        bias = self.params["bias"]
        in_shape = self.params["in_shape"]
        out_size = torch.prod(self.out_shape(in_shape))

        # Reshape to 2d, stacking the coefficients of the symbolic bound in dim 0, the "batch" dimension"
        x_2d = x.T.reshape((-1, *in_shape))

        # Perform convolution on the reshaped input
        y_2d = functional.conv2d(x_2d, weight=weights, stride=stride, padding=padding)

        # Add the bias to the last "batch" dimension, since this is the constant value of the bounds
        if add_bias:
            y_2d[-1, :, :, :] += bias.view(-1, 1, 1)

        # Reshape to NxM shaped where N are the nodes and M are the coefficients for the bounds
        y = y_2d.detach().reshape(-1, int(out_size)).T

        return y

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        if len(bounds_symbolic_pre) > 1:
            raise ValueError(f"Expected one set of symbolic bounds, got {len(bounds_symbolic_pre)}")
        bounds_symbolic_pre = bounds_symbolic_pre[0]

        stride = self.params["stride"]
        padding = self.params["padding"]
        weights = self.params["weight"]
        bias = self.params["bias"]
        groups = self.params["groups"]
        in_shape = self.params["in_shape"]
        out_size = torch.prod(self.out_shape(in_shape))

        pos_weigths, neg_weights = weights.clone(), weights.clone()
        pos_weigths[pos_weigths < 0] = 0
        neg_weights[neg_weights > 0] = 0

        # Reshape to 2d, stacking the coefficients of the symbolic bound in dim 0, the "batch" dimension"
        symb_low_pre = bounds_symbolic_pre[0].T.reshape((-1, *in_shape))
        symb_up_pre = bounds_symbolic_pre[1].T.reshape((-1, *in_shape))

        # Perform convolution on the reshaped input
        symb_low_post = functional.conv2d(symb_low_pre, weight=pos_weigths, stride=stride,
                                          padding=padding, groups=groups)
        symb_low_post += functional.conv2d(symb_up_pre, weight=neg_weights, stride=stride,
                                           padding=padding, groups=groups)
        symb_up_post = functional.conv2d(symb_low_pre, weight=neg_weights, stride=stride,
                                         padding=padding, groups=groups)
        symb_up_post += functional.conv2d(symb_up_pre, weight=pos_weigths, stride=stride,
                                          padding=padding, groups=groups)

        # Add the bias to the last "batch" dimension, since this is the constant value of the bounds
        if add_bias and bias is not None:

            symb_low_post[-1, :, :, :] += bias.view(-1, 1, 1)
            symb_up_post[-1, :, :, :] += bias.view(-1, 1, 1)

        # Reshape to NxM shaped where N are the nodes and M are the coefficients for the bounds
        symb_low_post = symb_low_post.reshape(-1, int(out_size)).T.unsqueeze(dim=0)
        symb_up_post = symb_up_post.reshape(-1, int(out_size)).T.unsqueeze(dim=0)

        return torch.cat((symb_low_post, symb_up_post), dim=0)

    # noinspection PyTypeChecker,PyCallingNonCallable
    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        in_shape = self.params["in_shape"]
        weights = self.params["weight"]

        bias = self.params["bias"]
        groups = self.params["groups"]
        kernel_size = self.params["kernel_size"]
        stride = self.params["stride"]
        padding = self.params["padding"]
        out_shape = self.out_shape(in_shape)

        num_eqs = bounds_symbolic_post.shape[0]

        transposed_conv_shape = ((out_shape[1] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                                 (out_shape[2] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])

        output_padding = (in_shape[1] - transposed_conv_shape[0], in_shape[2] - transposed_conv_shape[1])

        bounds_symbolic_pre = functional.conv_transpose2d(
            bounds_symbolic_post[:, :-1].reshape(bounds_symbolic_post.shape[0], *tuple(out_shape)), weight=weights,
            stride=stride, padding=padding, output_padding=output_padding, groups=groups).view(num_eqs, -1)

        bounds_symbolic_pre = torch.cat((bounds_symbolic_pre, bounds_symbolic_post[:, -1].unsqueeze(1)), dim=1)

        if add_bias and bias is not None:
            vars_pr_channel = torch.prod(self.out_shape(in_shape=torch.LongTensor(tuple(in_shape)))[1:3])
            new_bias = torch.sum(bounds_symbolic_post[:, :-1].view(num_eqs, -1, vars_pr_channel) *
                                 bias.view(1, -1, 1), dim=(1, 2))
            bounds_symbolic_pre[:, -1] += new_bias

        return [bounds_symbolic_pre]

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)

    # noinspection PyArgumentList,PyCallingNonCallable
    def out_shape(self, in_shape: torch.Tensor) -> torch.Tensor:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        params = self.params
        channels = params["out_channels"]
        height = int(in_shape[1] - params["kernel_size"][0] + 2 * params["padding"][0]) // int(params["stride"][0]) + 1
        width = int(in_shape[2] - params["kernel_size"][1] + 2 * params["padding"][1]) // int(params["stride"][1]) + 1

        return torch.LongTensor((channels, height, width))


class AvgPool2d(AbstractOperation):

    def __init__(self):

        self._out_unfolded = None
        self._weights = None

        super().__init__()

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return False

    @property
    def required_params(self) -> list:
        return ["kernel_size", "padding", "stride"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [nn.AvgPool2d]

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        raise NotImplementedError("Propagate not implemented")

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        if len(bounds_symbolic_pre) > 1:
            raise ValueError(f"Expected one set of symbolic bounds, got {len(bounds_symbolic_pre)}")
        bounds_symbolic_pre = bounds_symbolic_pre[0]

        stride = self.params["stride"]
        padding = self.params["padding"]
        in_shape = self.params["in_shape"]
        kernel_size = self.params["kernel_size"]
        out_size = torch.prod(self.out_shape(in_shape))

        # Reshape to 2d, stacking the coefficients of the symbolic bound in dim 0, the "batch" dimension".
        symb_low_pre = bounds_symbolic_pre[0].T.reshape((-1, *in_shape))
        symb_up_pre = bounds_symbolic_pre[1].T.reshape((-1, *in_shape))

        # Perform convolution on the reshaped input
        symb_low_post = functional.avg_pool2d(symb_low_pre, kernel_size=kernel_size, stride=stride, padding=padding)
        symb_up_post = functional.avg_pool2d(symb_up_pre, kernel_size=kernel_size, stride=stride, padding=padding)

        # Reshape to NxM shaped where N are the nodes and M are the coefficients for the bounds.
        symb_low_post = symb_low_post.reshape(-1, int(out_size)).T.unsqueeze(dim=0)
        symb_up_post = symb_up_post.reshape(-1, int(out_size)).T.unsqueeze(dim=0)

        return torch.cat((symb_low_post, symb_up_post), dim=0)

    # noinspection PyTypeChecker
    def rsip_backward(self,  bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        in_shape = self.params["in_shape"]
        kernel_size = self.params["kernel_size"]
        padding = self.params["padding"]
        stride = self.params["stride"]

        num_eqs = bounds_symbolic_post.shape[0]
        out_shape = self.out_shape(in_shape)

        if self._weights is None:
            self._weights = torch.zeros((out_shape[0], 1, *kernel_size),
                                        dtype=self._precision).to(device=bounds_symbolic_post.device)
            self._weights += 1/(kernel_size[0] * kernel_size[1])

        transposed_conv_shape = ((out_shape[1] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                                 (out_shape[2] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])

        output_padding = (in_shape[1] - transposed_conv_shape[0], in_shape[2] - transposed_conv_shape[1])

        bounds_symbolic_pre = functional.conv_transpose2d(
            bounds_symbolic_post[:, :-1].reshape(bounds_symbolic_post.shape[0], *tuple(out_shape)), self._weights,
            stride=stride, padding=padding, output_padding=output_padding, groups=in_shape[0]).view(num_eqs, -1)

        return [torch.cat((bounds_symbolic_pre, bounds_symbolic_post[:, -1].unsqueeze(1)), dim=1)]

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)

    # noinspection PyArgumentList,PyCallingNonCallable
    def out_shape(self, in_shape: torch.Tensor) -> torch.Tensor:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        params = self.params
        channels = in_shape[0]

        if isinstance(params["kernel_size"], int):
            params["kernel_size"] = (params["kernel_size"], params["kernel_size"])
        if isinstance(params["padding"], int):
            params["padding"] = (params["padding"], params["padding"])
        if isinstance(params["stride"], int):
            params["stride"] = (params["stride"], params["stride"])

        height = int(in_shape[1] - params["kernel_size"][0] + 2 * params["padding"][0]) // int(params["stride"][0]) + 1
        width = int(in_shape[2] - params["kernel_size"][1] + 2 * params["padding"][1]) // int(params["stride"][1]) + 1

        return torch.LongTensor((channels, height, width))


class Mean(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return False

    @property
    def required_params(self) -> list:
        return ["dims", "keepdim"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [MeanTorch]

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        raise NotImplementedError("Propagate not implemented")

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        if len(bounds_symbolic_pre) > 1:
            raise ValueError(f"Expected one set of symbolic bounds, got {len(bounds_symbolic_pre)}")
        bounds_symbolic_pre = bounds_symbolic_pre[0]

        in_shape = self.params["in_shape"]

        if 0 in self.params["dims"]:
            raise ValueError("Mean over batch dimension is not supported")

        out_size = torch.prod(self.out_shape(in_shape))

        # Reshape to 2d, stacking the coefficients of the symbolic bound in dim 0, the "batch" dimension"
        symb_low_pre = bounds_symbolic_pre[0].T.reshape((-1, *in_shape))
        symb_up_pre = bounds_symbolic_pre[1].T.reshape((-1, *in_shape))

        # Perform convolution on the reshaped input
        symb_low_post = torch.mean(symb_low_pre, dim=self.params["dims"], keepdim=self.params["keepdim"])
        symb_up_post = torch.mean(symb_up_pre, dim=self.params["dims"], keepdim=self.params["keepdim"])

        # Reshape to NxM shaped where N are the nodes and M are the coefficients for the bounds.
        symb_low_post = symb_low_post.reshape(-1, int(out_size)).T.unsqueeze(dim=0)
        symb_up_post = symb_up_post.reshape(-1, int(out_size)).T.unsqueeze(dim=0)

        return torch.cat((symb_low_post, symb_up_post), dim=0)

    # noinspection PyTypeChecker
    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        in_shape = self.params["in_shape"]
        dims = self.params["dims"]
        num_eqs = bounds_symbolic_post.shape[0]

        assert dims == (2, 3), "Error, Mean only implemented for dims = (2,3)"

        bounds_symbolic_pre = torch.zeros((bounds_symbolic_post.shape[0], *in_shape),
                                          dtype=self._precision).to(device=bounds_symbolic_post.device)
        divisor = (in_shape[1] * in_shape[2])

        bounds_symbolic_pre[:, :] = bounds_symbolic_post[:, :-1].view(num_eqs, -1, 1, 1) / divisor
        bounds_symbolic_pre = torch.cat((bounds_symbolic_pre.view((num_eqs, -1)),
                                        bounds_symbolic_post[:, -1].view(-1, 1)), dim=1)

        return [bounds_symbolic_pre]

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)

    # noinspection PyArgumentList,PyCallingNonCallable
    def out_shape(self, in_shape: torch.Tensor) -> torch.Tensor:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        out_shape = []
        for i in range(len(in_shape)):
            if i+1 not in self.params["dims"]:
                out_shape.append(in_shape[i])
            else:
                if self.params["keepdim"]:
                    out_shape.append(1)

        return torch.LongTensor(out_shape)


class MulConstant(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return False

    @property
    def required_params(self) -> list:
        return ["multiplier"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [MulConstantTorch]

    def process_params(self):

        """
        Removes batch dimension from 'term' if present.
        """

        if len(self.params["multiplier"].shape) == len(self.params["in_shape"]) + 1:

            if self.params["multiplier"].shape[0] == 1:
                self.params["multiplier"] = self.params["multiplier"][0]  # Remove batch dimension.
            else:
                raise ValueError("Unexpected shape for multiplier in AddConstant.")

        elif (len(self.params["multiplier"].shape) != 1 and
              len(self.params["multiplier"].shape) == len(self.params["in_shape"])):

            raise ValueError("Unexpected shape for multiplier in AddConstant.")

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        bounds_shape = x.shape
        org_shape = (*self.params["in_shape"], x.shape[1])
        return (x.reshape(org_shape) * self.params["multiplier"]).reshape(bounds_shape)

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        if len(bounds_symbolic_pre) > 1:
            raise ValueError(f"Expected one set of symbolic bounds, got {len(bounds_symbolic_pre)}")

        bounds_symbolic_pre = bounds_symbolic_pre[0]
        mul = self.params["multiplier"].unsqueeze(0).unsqueeze(-1)

        bounds_shape = bounds_symbolic_pre.shape
        org_shape = (bounds_symbolic_pre.shape[0], *self.params["in_shape"], bounds_symbolic_pre.shape[2])

        if torch.sum(mul < 0) == 0:
            bounds_symbolic_post = (mul * bounds_symbolic_pre.reshape(org_shape)).reshape(bounds_shape)
        elif sum(mul.reshape(-1).shape) == 1 and mul.reshape(-1) > 0:
            bounds_symbolic_post = (mul.reshape(-1) * bounds_symbolic_pre.reshape(org_shape)).reshape(bounds_shape)
        elif sum(mul.reshape(-1).shape) == 1 and mul.reshape(-1) <= 0:
            bounds_symbolic_post = (mul.reshape(-1) * torch.flip(bounds_symbolic_pre,
                                                                 (0,)).reshape(org_shape)).reshape(bounds_shape)
        else:
            raise ValueError("Negative values in mul not supported for multidimensional multiplier")
        return bounds_symbolic_post

    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        bounds_shape = bounds_symbolic_post.shape
        org_shape = (bounds_symbolic_post.shape[0], *self.params["in_shape"])
        bounds_symbolic_pre = bounds_symbolic_post
        new_bounds = bounds_symbolic_pre[:, :-1].reshape(org_shape) * self.params["multiplier"]
        bounds_symbolic_pre[:, :-1] = new_bounds.reshape((bounds_shape[0], bounds_shape[1] - 1))

        return [bounds_symbolic_pre]

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)


class AddConstant(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return True

    @property
    def required_params(self) -> list:
        return ["term"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [AddConstantTorch]

    def process_params(self):

        """
        Removes batch dimension from 'term' if present.
        """

        if len(self.params["term"].shape) == len(self.params["in_shape"]) + 1:

            if self.params["term"].shape[0] == 1:
                self.params["term"] = self.params["term"][0]  # Remove batch dimension.
            else:
                raise ValueError("Unexpected shape for term in AddConstant.")

        elif (len(self.params["term"].shape) != 1 and
                len(self.params["term"].shape) == len(self.params["in_shape"])):

            raise ValueError("Unexpected shape for term in AddConstant.")

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        bounds_shape = x.shape
        org_shape = (*self.params["in_shape"], x.shape[1])
        return (x.reshape(org_shape) + self.params["term"]).reshape(bounds_shape)

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        if len(bounds_symbolic_pre) > 1:
            raise ValueError(f"Expected one set of symbolic bounds, got {len(bounds_symbolic_pre)}")

        bounds_symbolic_pre = bounds_symbolic_pre[0]
        term = self.params["term"].unsqueeze(0)

        org_shape = (bounds_symbolic_pre.shape[0], *self.params["in_shape"])

        bounds_symbolic_post = bounds_symbolic_pre.clone()
        bounds_symbolic_post[:, :, -1] = (bounds_symbolic_post[:, :, -1].reshape(org_shape) + term).reshape(2, -1)

        return bounds_symbolic_post

    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        term = self.params["term"]

        bounds_symbolic_pre = bounds_symbolic_post
        new_constants = (torch.zeros(tuple(self.params["in_shape"]), dtype=self._precision) + term).reshape(1, -1)
        bounds_symbolic_pre[:, -1] += torch.sum(bounds_symbolic_pre[:, :-1] * new_constants, dim=1)

        return [bounds_symbolic_pre]

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)


class AddDynamic(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return True

    @property
    def required_params(self) -> list:
        return []

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [AddDynamicTorch]

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        return x[0] + x[1]

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        return bounds_symbolic_pre[0] + bounds_symbolic_pre[1]

    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        bounds_symbolic_pre = [bounds_symbolic_post, bounds_symbolic_post.clone()]
        bounds_symbolic_pre[1][:, -1] = 0

        return bounds_symbolic_pre

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)


class Crop(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return True

    @property
    def required_params(self) -> list:
        return ["crop"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [CropTorch]

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        in_shape = self.params["in_shape"]
        crop = self.params["crop"]
        num_eqs = x.shape[0]

        x = x.view((num_eqs, *in_shape))
        x = x[..., crop:-crop, crop:-crop]

        return x.reshape(num_eqs, -1)

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the symbolic bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        if len(bounds_symbolic_pre) > 1:
            raise ValueError(f"Expected one set of symbolic bounds, got {len(bounds_symbolic_pre)}")
        bounds_symbolic_pre = bounds_symbolic_pre[0]

        in_shape = self.params["in_shape"]
        crop = self.params["crop"]
        num_eqs = bounds_symbolic_pre.shape[1]

        bounds_symbolic_post = bounds_symbolic_pre[:, :, :-1].view((2, num_eqs, *in_shape))
        bounds_symbolic_post = bounds_symbolic_post[..., crop:-crop, crop:-crop]

        return torch.cat((bounds_symbolic_post.reshape(2, num_eqs, -1),
                          bounds_symbolic_pre[:, :, -1].unsqueeze(2)), dim=2)

    # noinspection PyTypeChecker
    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        in_shape = self.params["in_shape"]
        out_shape = self.out_shape(in_shape)
        crop = self.params["crop"]
        num_eqs = bounds_symbolic_post.shape[0]

        bounds_symbolic_pre = bounds_symbolic_post[:, :-1].view((num_eqs, *out_shape))
        bounds_symbolic_pre = functional.pad(bounds_symbolic_pre, (crop, crop, crop, crop), "constant", 0)

        return [torch.cat((bounds_symbolic_pre.view(num_eqs, -1), bounds_symbolic_post[:, -1].unsqueeze(1)), dim=1)]

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)

    # noinspection PyCallingNonCallable
    def out_shape(self, in_shape: torch.Tensor) -> torch.Tensor:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        crop = self.params["crop"]

        return torch.LongTensor((*in_shape[:-2], in_shape[-2] - 2*crop, in_shape[-1] - 2*crop))


class Unsqueeze(AbstractOperation):

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_monotonically_increasing(self) -> bool:
        return True

    @property
    def required_params(self) -> list:
        return ["dims"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current
            subclass.
        """

        return [UnsqueezeTorch]

    def forward(self, x: torch.Tensor, add_bias: bool = True) -> torch.Tensor:

        """
        Propagates through the operation.

        Args:
            x:
                The input as a NxM tensor where each row is a symbolic bounds for
                the corresponding node. Can be used on concrete values instead of
                bound by shaping them into a Nx1 tensor.
            add_bias:
                If true, the bias is added.
        Returns:
            The value of the activation function at const_terms.
        """

        return x

    def ssip_forward(self, bounds_symbolic_pre: list, add_bias: bool = True,
                     calc_nodes: torch.Tensor = None) -> torch.Tensor:

        """
        Propagates the upper and lower symbolic bounds of ssip.

        Args:
            bounds_symbolic_pre:
                A list of 2xNxM tensor with the symbolic bounds. The list contains
                all input bounds, the first dimension of the bounds contains
                the lower and upper bounds respectively.
            add_bias:
                If true, the bias is added.
            calc_nodes:
                If provided, only these output nodes are calculated.
        Returns:
            The post-op values
        """

        return bounds_symbolic_pre[0]

    def rsip_backward(self, bounds_symbolic_post: torch.Tensor, add_bias: bool = True) -> list:

        """
        Reverse propagates the given symbolic bounds by substituting variables from the
        previous node. Used by the RSIP algorithm.

        Args:
            bounds_symbolic_post:
                A 2d array where each row is an equation, the columns are coefficients
                and the last element in each row is the constant of the symbolic bound.
            add_bias:
                If true, the bias is considered.
        Returns:
            The new bound with respect to the variables from the preceding node.
        """

        return [bounds_symbolic_post]

    # noinspection PyTypeChecker
    def linear_relaxation(self, lower_bounds_concrete_in: torch.Tensor, upper_bounds_concrete_in: torch.Tensor,
                          force_parallel: bool = False) -> torch.Tensor:

        """
        Not implemented for linear activations.
        """

        msg = f"linear_relaxation(...) not implemented"
        raise NotImplementedError(msg)

    # noinspection PyArgumentList,PyCallingNonCallable,PyTypeChecker
    def out_shape(self, in_shape: torch.Tensor) -> torch.Tensor:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        dims = [dim - 1 for dim in self.params["dims"]]  # No batch dim in sip

        new_shape = [1] * (len(dims) + len(in_shape))

        j = 0
        for i in range(len(new_shape)):

            if i in dims:
                continue
            else:
                new_shape[i] = in_shape[j]
                j += 1

        return torch.LongTensor(new_shape)
