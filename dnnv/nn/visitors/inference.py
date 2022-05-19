from __future__ import annotations

from typing import Dict

import numpy as np

from .. import Operation, operations
from ..utils import TensorDetails
from .base import OperationVisitor


# TODO: need to implement methods for all Operations types
class DetailsInference(OperationVisitor):
    def __init__(self):
        super().__init__(recursive=False)
        self.tensor_details: Dict[Operation, TensorDetails] = {}

    def generic_visit(self, operation):
        if isinstance(operation, Operation):
            return super().generic_visit(operation)
        elif isinstance(operation, np.ndarray):
            self.tensor_details[operation] = TensorDetails(
                operation.shape, operation.dtype
            )
        return super().generic_visit(operation)

    def get_details(self, obj):
        if isinstance(obj, Operation):
            self.visit(obj)
            return self.tensor_details[obj]
        elif isinstance(obj, np.ndarray):
            return TensorDetails(obj.shape, obj.dtype)
        elif isinstance(obj, np.number):
            return TensorDetails((), np.dtype(type(obj)))
        raise TypeError(f"Unknown tensor type: {type(obj)}")

    def visit_Add(self, operation: operations.Add):
        a = operation.a
        if isinstance(a, Operation):
            self.visit(a)
            a_details = self.tensor_details[a]
            a = np.zeros(a_details.shape, dtype=a_details.dtype)
        b = operation.b
        if isinstance(b, Operation):
            self.visit(b)
            b_details = self.tensor_details[b]
            b = np.zeros(b_details.shape, dtype=b_details.dtype)
        output = a + b
        self.tensor_details[operation] = TensorDetails(output.shape, output.dtype)

    def visit_BatchNormalization(self, operation: operations.BatchNormalization):
        self.tensor_details[operation] = self.get_details(operation.x)

    def visit_Concat(self, operation: operations.Concat):
        axis = operation.axis
        output_shape = None
        output_dtype = None
        for x in operation.x:
            shape, dtype = self.get_details(x)
            if output_shape is None:
                output_shape = list(shape)
                output_dtype = dtype
            else:
                output_shape[axis] += shape[axis]
        self.tensor_details[operation] = TensorDetails(
            tuple(output_shape), output_dtype
        )

    def visit_Conv(self, operation: operations.Conv):
        x_details = self.get_details(operation.x)
        batch, in_c, in_h, in_w = x_details.shape
        output_dtype = x_details.dtype
        w = operation.w
        assert isinstance(w, np.ndarray)
        b = operation.b
        assert b is None or isinstance(b, np.ndarray)
        assert np.all(operation.group == 1)
        assert np.all(operation.dilations == 1)

        batch = batch if batch >= 0 else 1
        k_h, k_w = w.shape[2:4]
        p_top, p_left, p_bottom, p_right = operation.pads
        s_h, s_w = operation.strides

        out_c = w.shape[0]
        out_h = (in_h - k_h + p_top + p_bottom) // s_h + 1
        out_w = (in_w - k_w + p_left + p_right) // s_w + 1
        output_shape = (batch, out_c, out_h, out_w)

        self.tensor_details[operation] = TensorDetails(output_shape, output_dtype)

    def visit_Div(self, operation: operations.Div):
        a = operation.a
        if isinstance(a, Operation):
            self.visit(a)
            a_details = self.tensor_details[a]
            a = np.zeros(a_details.shape, dtype=a_details.dtype)
        b = operation.b
        if isinstance(b, Operation):
            self.visit(b)
            b_details = self.tensor_details[b]
            b = np.zeros(b_details.shape, dtype=b_details.dtype)
        output = a / b
        self.tensor_details[operation] = TensorDetails(output.shape, output.dtype)

    def visit_Elu(self, operation: operations.Elu):
        self.tensor_details[operation] = self.get_details(operation.x)

    def visit_Flatten(self, operation: operations.Flatten):
        axis = operation.axis
        input_details = self.get_details(operation.x)
        input_shape = input_details.shape
        output_shape = (
            int(np.product(input_shape[:axis])),
            int(np.product(input_shape[axis:])),
        )
        self.tensor_details[operation] = TensorDetails(
            output_shape, input_details.dtype
        )

    def visit_Gather(self, operation: operations.Gather):
        assert isinstance(operation.indices, np.ndarray)
        if operation.indices.size == 1:
            indices = ()
        else:
            indices = tuple(operation.indices)
        axis = operation.axis
        input_details = self.get_details(operation.x)
        self.tensor_details[operation] = TensorDetails(
            input_details.shape[:axis] + indices + input_details.shape[axis + 1 :],
            input_details.dtype,
        )

    def visit_Gemm(self, operation: operations.Gemm):
        a = operation.a
        if isinstance(a, Operation):
            self.visit(a)
            a_details = self.tensor_details[a]
            a = np.zeros(a_details.shape, dtype=a_details.dtype)
        if operation.transpose_a:
            a = a.T
        b = operation.b
        if isinstance(b, Operation):
            self.visit(b)
            b_details = self.tensor_details[b]
            b = np.zeros(b_details.shape, dtype=b_details.dtype)
        if operation.transpose_b:
            b = b.T
        c = operation.c
        if isinstance(c, Operation):
            self.visit(c)
            c_details = self.tensor_details[c]
            c = np.zeros(c_details.shape, dtype=c_details.dtype)
        output = a @ b + c
        self.tensor_details[operation] = TensorDetails(output.shape, output.dtype)

    def visit_Input(self, operation: operations.Input):
        self.tensor_details[operation] = TensorDetails(
            tuple(d if d >= 0 else 1 for d in operation.shape), operation.dtype
        )

    def visit_LeakyRelu(self, operation: operations.LeakyRelu):
        self.tensor_details[operation] = self.get_details(operation.x)

    def visit_MatMul(self, operation: operations.MatMul):
        a = operation.a
        if isinstance(a, Operation):
            self.visit(a)
            a_details = self.tensor_details[a]
            a = np.zeros(a_details.shape, dtype=a_details.dtype)
        b = operation.b
        if isinstance(b, Operation):
            self.visit(b)
            b_details = self.tensor_details[b]
            b = np.zeros(b_details.shape, dtype=b_details.dtype)
        output = a @ b
        self.tensor_details[operation] = TensorDetails(output.shape, output.dtype)

    def visit_MaxPool(self, operation: operations.MaxPool):
        x_details = self.get_details(operation.x)
        batch_size, num_channels, *spatial_dims = x_details.shape
        output_dtype = x_details.dtype

        num_spatial_dims = len(spatial_dims)
        batch_size = batch_size if batch_size >= 0 else 1
        output_spatial_dims = [0 for _ in range(num_spatial_dims)]
        pad_shape = [
            operation.pads[i] + operation.pads[i + num_spatial_dims]
            for i in range(num_spatial_dims)
        ]

        if operation.ceil_mode:
            truncate_op = np.ceil
        else:
            truncate_op = np.floor
        for i in range(num_spatial_dims):
            output_spatial_dims[i] = truncate_op(
                (
                    spatial_dims[i]
                    + pad_shape[i]
                    - ((operation.kernel_shape[i] - 1) * operation.dilations[i] + 1)
                )
                / operation.strides[i]
                + 1
            )
        self.tensor_details[operation] = TensorDetails(
            (batch_size, num_channels, *output_spatial_dims),
            output_dtype,
        )

    def visit_Mul(self, operation: operations.Mul):
        a = operation.a
        if isinstance(a, Operation):
            self.visit(a)
            a_details = self.tensor_details[a]
            a = np.zeros(a_details.shape, dtype=a_details.dtype)
        b = operation.b
        if isinstance(b, Operation):
            self.visit(b)
            b_details = self.tensor_details[b]
            b = np.zeros(b_details.shape, dtype=b_details.dtype)
        output = a * b
        self.tensor_details[operation] = TensorDetails(output.shape, output.dtype)

    def visit_RandomUniform(self, operation: operations.RandomUniform):
        self.tensor_details[operation] = TensorDetails(
            tuple(d for d in operation.shape), operation.dtype
        )

    def visit_Relu(self, operation: operations.Relu):
        self.tensor_details[operation] = self.get_details(operation.x)

    def visit_Reshape(self, operation: operations.Reshape):
        self.visit(operation.x)
        input_details = self.tensor_details[operation.x]
        if not isinstance(operation.shape, Operation):
            new_shape = np.empty(input_details.shape).reshape(operation.shape).shape
            self.tensor_details[operation] = TensorDetails(
                new_shape, input_details.dtype
            )
            return
        self.visit(operation.shape)
        if (
            self.tensor_details[operation.shape].shape == (2,)
            and isinstance(operation.shape, operations.Concat)
            and isinstance(operation.shape.x[1], np.ndarray)
            and operation.shape.x[1].size == 1
            and operation.shape.x[1].item() == -1
            and isinstance(operation.shape.x[0], operations.Unsqueeze)
            and np.all(operation.shape.x[0].axes == 0)
        ):
            new_shape = np.empty(input_details.shape).reshape(1, -1).shape
            self.tensor_details[operation] = TensorDetails(
                new_shape, input_details.dtype
            )
            return
        raise NotImplementedError()

    def visit_Shape(self, operation: operations.Shape):
        input_details = self.get_details(operation.x)
        self.tensor_details[operation] = TensorDetails(
            np.shape(input_details.shape), np.dtype(np.int64)
        )

    def visit_Sigmoid(self, operation: operations.Sigmoid):
        self.tensor_details[operation] = self.get_details(operation.x)

    def visit_Softmax(self, operation: operations.Softmax):
        self.tensor_details[operation] = self.get_details(operation.x)

    def visit_Sub(self, operation: operations.Sub):
        a = operation.a
        if isinstance(a, Operation):
            self.visit(a)
            a_details = self.tensor_details[a]
            a = np.zeros(a_details.shape, dtype=a_details.dtype)
        b = operation.b
        if isinstance(b, Operation):
            self.visit(b)
            b_details = self.tensor_details[b]
            b = np.zeros(b_details.shape, dtype=b_details.dtype)
        output = a - b
        self.tensor_details[operation] = TensorDetails(output.shape, output.dtype)

    def visit_Tanh(self, operation: operations.Tanh):
        self.tensor_details[operation] = self.get_details(operation.x)

    def visit_Transpose(self, operation: operations.Transpose):
        input_details = self.get_details(operation.x)
        new_shape = tuple(np.array(input_details.shape)[operation.permutation])
        self.tensor_details[operation] = TensorDetails(new_shape, input_details.dtype)

    def visit_Unsqueeze(self, operation: operations.Unsqueeze):
        assert not isinstance(operation.axes, operations.Operation)
        axes = tuple(operation.axes)
        input_details = self.get_details(operation.x)
        new_shape = [0 for _ in range(len(axes) + len(input_details.shape))]
        for a in axes:
            new_shape[a] = 1
        for d in input_details.shape:
            new_shape[new_shape.index(0)] = d
        self.tensor_details[operation] = TensorDetails(
            tuple(new_shape),
            input_details.dtype,
        )


__all__ = ["DetailsInference"]
