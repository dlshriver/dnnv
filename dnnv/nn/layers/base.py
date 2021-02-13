"""
"""
import numpy as np

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional, Type, Union

from .. import OperationGraph
from ..operations import *
from ..visitors import OperationCounter
from ..transformers import DropPrefix
from ...utils import get_subclasses


class _Layer(type):
    def __new__(self, name, bases, namespace, **kwargs):
        if name == "Layer":
            return super().__new__(self, name, bases, namespace, **kwargs)
        if "OP_PATTERN" not in namespace:
            raise TypeError(f"Layer {name} must specify `OP_PATTERN`")
        op_pattern = namespace["OP_PATTERN"]
        if (
            op_pattern is not None
            and not isinstance(op_pattern, OperationPattern)
            and (
                not isinstance(op_pattern, type)
                or not issubclass(op_pattern, Operation)
            )
        ):
            raise TypeError("`OP_PATTERN` must be an operation pattern")
        return super().__new__(self, name, bases, namespace, **kwargs)

    @property
    def OP_PATTERN(self) -> Union[Type[Operation], OperationPattern, None]:
        return self.__dict__["OP_PATTERN"]


class LayerMatch:
    def __init__(self, layer, input_op_graph):
        self.layer = layer
        self.input_op_graph = input_op_graph


class Layer(metaclass=_Layer):
    @classmethod
    @abstractmethod
    def from_operation_graph(cls, operation_graph):
        raise NotImplementedError()

    @classmethod
    def match(
        cls: Type["Layer"],
        operation_graph: OperationGraph,
        layer_types: Optional[List[Type["Layer"]]] = None,
    ) -> Optional[LayerMatch]:
        if cls is Layer and layer_types is None:
            layer_types = list(get_subclasses(cls))
        elif cls is not Layer:
            if layer_types is not None:
                raise TypeError(
                    "match() got an unexpected keyword argument 'layer_types'"
                )
            layer_types = [cls]

        best_match: Optional[List[Operation]] = None
        best_op_count = float("inf")
        best_layer_type = Layer
        assert layer_types is not None
        for layer_type in layer_types:
            if layer_type.OP_PATTERN is None:
                continue
            matches = layer_type.OP_PATTERN.match(operation_graph.output_operations)
            for match in matches:
                op_count = 0
                visitor = OperationCounter()
                for op in match:
                    op_count = visitor.visit(op)
                if op_count < best_op_count:
                    best_match = match
                    best_op_count = op_count
                    best_layer_type = layer_type
        if best_match is None:
            return None
        input_op_graph = OperationGraph(best_match)
        op_graph = OperationGraph(operation_graph.walk(DropPrefix(input_op_graph)))
        return LayerMatch(
            best_layer_type.from_operation_graph(op_graph), input_op_graph
        )


class InputLayer(Layer):
    OP_PATTERN = Input

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    @classmethod
    def from_operation_graph(cls, operation_graph):
        shape = tuple(
            d if d >= 0 else 1 for d in operation_graph.output_operations[0].shape
        )
        dtype = operation_graph.output_operations[0].dtype
        return cls(shape, dtype)


class FullyConnected(Layer):
    OP_PATTERN = (
        (((Transpose | None) >> (Flatten | Reshape)) | None)
        >> (Gemm | (MatMul >> Add))
        >> (Activation | None)
    )

    def __init__(self, weights, bias, activation=None, w_permutation=None):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.w_permutation = (
            w_permutation
            if w_permutation is not None
            else np.arange(self.weights.shape[0])
        )

    @classmethod
    def from_operation_graph(cls, operation_graph):
        # check activation type
        op = operation_graph.output_operations
        assert len(op) == 1
        op = op[0]
        activation = None
        if isinstance(op, (Relu, Sigmoid, Tanh)):
            activation = op.__class__.__name__.lower()
            op = op.inputs
            assert len(op) == 1
            op = op[0]
        elif not isinstance(op, (Gemm, Add)):
            raise ValueError(
                "Expected operation of type (Gemm | Add | Activation), but got %s"
                % op.__class__.__name__
            )

        # get weights and biases
        weights = None
        bias = None
        if isinstance(op, Gemm):
            if op.alpha != 1.0 or op.beta != 1.0:
                raise ValueError("Scaling not supported in Fully Connected layers.")
            if not isinstance(op.a, Operation):
                raise ValueError(
                    "Constant input tensors are not supported for GeMM "
                    "in Fully Connected layers."
                )
            if op.transpose_a:
                raise ValueError(
                    "Transposing input to Fully Connected layer is not supported."
                )
            if isinstance(op.b, Operation):
                raise ValueError(
                    "Multiple input tensors are not supported for GeMM "
                    "in Fully Connected layers."
                )
            weights = op.b
            if op.transpose_b:
                weights = weights.T
            if isinstance(op.c, Operation):
                raise ValueError(
                    "Variable input tensors are not supported for GeMM bias "
                    "in Fully Connected layers."
                )
            bias = op.c
        elif isinstance(op, Add):
            if not isinstance(op.a, Operation):
                raise ValueError(
                    "Constant input tensors are not supported for Add "
                    "in Fully Connected layers."
                )
            bias = op.b
            op = op.a
            if not isinstance(op.a, Operation):
                raise ValueError(
                    "Constant input tensors are not supported for MatMul "
                    "in Fully Connected layers."
                )
            weights = op.b
        else:
            raise ValueError(
                "Expected type (Gemm | (MatMul >> Add)), but got %s"
                % op.__class__.__name__
            )

        op = op.inputs
        assert len(op) == 1
        op = op[0]
        if isinstance(op, Input):
            return cls(weights, bias, activation=activation)
        if not isinstance(op, (Flatten, Reshape)):
            raise ValueError(
                "Expected type (None | (Transpose >> (Flatten | Reshape))), but got %s"
                % op.__class__.__name__
            )
        op = op.inputs
        if len(op) > 1:
            return cls(weights, bias, activation=activation)
        assert len(op) == 1
        op = op[0]
        if isinstance(op, Input):
            return cls(weights, bias, activation=activation)
        elif isinstance(op, Transpose):
            if not isinstance(op.x, Input):
                raise ValueError("Expected Transpose to be applied to Input.")
            permutation = np.asarray(op.permutation)
            undo_permutation = permutation[permutation]
            input_shape = np.asarray(op.x.shape)[permutation]
            weights_permutation = (
                np.arange(np.product(input_shape))
                .reshape(input_shape)
                .transpose(undo_permutation)
                .flatten()
            )
        else:
            raise ValueError(
                "Expected type Transpose, but got %s" % op.__class__.__name__
            )
        return cls(
            weights, bias, activation=activation, w_permutation=weights_permutation
        )


class Convolutional(Layer):
    OP_PATTERN = Conv >> (Activation | None)

    def __init__(
        self, weights, bias, activation=None, kernel_shape=None, strides=1, pads=0
    ):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.kernel_shape = kernel_shape
        if self.kernel_shape is None:
            self.kernel_shape = self.weights.shape[2:]
        self.strides = strides
        self.pads = pads

    @classmethod
    def from_operation_graph(cls, operation_graph):
        op = operation_graph.output_operations
        assert len(op) == 1
        op = op[0]

        # check activation type
        activation = None
        if isinstance(op, (Relu, Sigmoid, Tanh)):
            activation = op.__class__.__name__.lower()
            op = op.inputs
            assert len(op) == 1
            op = op[0]
        elif not isinstance(op, Conv):
            raise ValueError(
                "Expected operation of type (Conv | Activation), but got %s"
                % op.__class__.__name__
            )

        # get weights, biases, and configuration
        weights = None
        bias = None
        kernel_shape = None
        pads = None
        strides = None
        if isinstance(op, Conv):
            if np.any(op.dilations != 1):
                raise ValueError(
                    "Dilation is currently not supported in Convolutional layers."
                )
            if op.group != 1:
                raise ValueError(
                    "Grouping is currently not supported in Convolutional layers."
                )
            if not isinstance(op.x, Operation):
                raise ValueError(
                    "Constant input tensors are not supported for Conv "
                    "in Convolutional layers."
                )
            weights = op.w
            bias = op.b
            kernel_shape = op.kernel_shape
            strides = op.strides
            pads = op.pads
        else:
            raise ValueError("Expected type Conv, but got %s" % op.__class__.__name__)
        return cls(
            weights,
            bias,
            activation=activation,
            kernel_shape=kernel_shape,
            strides=strides,
            pads=pads,
        )
