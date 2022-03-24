from __future__ import annotations

import numpy as np

from typing import Dict

from .base import OperationVisitor
from .. import operations, Operation
from ..utils import TensorDetails


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

    def visit_Concat(self, operation: operations.Concat):
        self.visit(operation.x[0])
        axis = operation.axis
        details = self.tensor_details[operation.x[0]]
        shape = list(details.shape)
        dtype = details.dtype
        for x in operation.x[1:]:
            self.visit(x)
            details = self.tensor_details[x]
            shape[axis] += details.shape[axis]
        self.tensor_details[operation] = TensorDetails(tuple(shape), dtype)

    def visit_Conv(self, operation: operations.Conv):
        x = operation.x
        if isinstance(x, Operation):
            self.visit(x)
            x_details = self.tensor_details[operation.x]
            batch, in_c, in_h, in_w = x_details.shape
            output_dtype = x_details.dtype
        else:
            assert isinstance(x, np.ndarray)
            batch, in_c, in_h, in_w = x.shape
            output_dtype = x.dtype
        w = operation.w
        assert isinstance(w, np.ndarray)
        b = operation.b
        assert isinstance(b, np.ndarray)
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
        self.visit(operation.x)
        self.tensor_details[operation] = self.tensor_details[operation.x]

    def visit_Flatten(self, operation: operations.Flatten):
        self.visit(operation.x)
        axis = operation.axis
        input_details = self.tensor_details[operation.x]
        input_shape = input_details.shape
        output_shape = (
            int(np.product(input_shape[:axis])),
            int(np.product(input_shape[axis:])),
        )
        self.tensor_details[operation] = TensorDetails(
            output_shape, input_details.dtype
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
        self.visit(operation.x)
        self.tensor_details[operation] = self.tensor_details[operation.x]

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
        self.visit(operation.x)
        self.tensor_details[operation] = self.tensor_details[operation.x]

    def visit_Sigmoid(self, operation: operations.Sigmoid):
        self.visit(operation.x)
        self.tensor_details[operation] = self.tensor_details[operation.x]

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
        self.visit(operation.x)
        self.tensor_details[operation] = self.tensor_details[operation.x]


__all__ = ["DetailsInference"]
