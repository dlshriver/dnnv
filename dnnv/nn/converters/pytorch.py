import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from .. import operations
from ..graph import OperationGraph
from ..operations import Operation
from ..visitors import OperationVisitor


def convert(op_graph: OperationGraph):
    converter = PytorchConverter(op_graph)
    model = converter.convert()

    return model


class PytorchOperation:
    def __init__(
        self,
        name: str,
        module: Optional[nn.Module],
        inputs: List[str] = [],
        outputs: List[str] = [],
    ):
        assert len(outputs) == 1
        super().__init__()
        self.name = name
        self.module = module
        self.inputs = inputs
        self.outputs = outputs


class PytorchModel(nn.Module):
    def __init__(
        self, operations: List[PytorchOperation], inputs: List[str], outputs: List[str]
    ):
        super().__init__()
        self.graph = {op.name: op for op in operations}
        self.module_dict = nn.ModuleDict(
            {op.name: op.module for op in operations if op.module}
        )
        self.inputs = inputs
        self.outputs = outputs

    def forward(self, *x):
        assert len(x) == len(self.inputs)
        cache = {n: x_ for n, x_ in zip(self.inputs, x)}
        computation_stack = [n for n in reversed(self.outputs)]
        while computation_stack:
            name = computation_stack.pop()
            if all(input_name in cache for input_name in self.graph[name].inputs):
                cache[self.graph[name].outputs[0]] = self.module_dict[name](
                    *[cache[input_name] for input_name in self.graph[name].inputs]
                )
            else:
                computation_stack.append(name)
                for input_name in self.graph[name].inputs:
                    if input_name not in cache:
                        computation_stack.append(input_name)
        return tuple(cache[output_name] for output_name in self.outputs)


class Flatten(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return x.flatten(self.axis)


class Permute(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x, permutation=None):
        if permutation is None:
            return x.permute(*self.permutation)
        return x.permute(*permutation)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x, shape=None):
        if shape is None:
            return x.view(*self.shape)
        return x.view(*shape)


class PytorchConverter(OperationVisitor):
    def __init__(self, op_graph: OperationGraph):
        self.op_graph = op_graph
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.visited: Dict[Operation, PytorchOperation] = {}
        self.op_counts: Dict[str, int] = defaultdict(int)

    def convert(self) -> nn.Module:
        output_details = self.op_graph.output_details
        for op, (shape, dtype) in zip(self.op_graph.output_operations, output_details):
            output_op = self.visit(op)
            self.outputs.append(output_op.name)

        model = PytorchModel(
            list(self.visited.values()), inputs=self.inputs, outputs=self.outputs
        )

        return model

    def _get_name(self, operation: Operation) -> str:
        optype = str(type(operation))
        idx = self.op_counts[optype] = self.op_counts[optype] + 1
        name = f"{optype}_{idx}"
        return name

    def visit(self, operation: Operation) -> PytorchOperation:
        if operation not in self.visited:
            result = super().visit(operation)
            self.visited[operation] = result
        return self.visited[operation]

    def generic_visit(self, operation: Operation):
        if not hasattr(self, "visit_%s" % operation.__class__.__name__):
            raise ValueError(
                "PyTorch converter not implemented for operation type %s"
                % operation.__class__.__name__
            )
        return super().generic_visit(operation)

    def visit_Conv(self, operation: operations.Conv) -> PytorchOperation:
        opname = self._get_name(operation)

        x = operation.x
        if isinstance(x, Operation):
            x = self.visit(x)

        in_c = operation.w.shape[1]
        out_c = operation.b.shape[0]
        pad_top, pad_left, pad_bottom, pad_right = operation.pads

        if pad_top == pad_bottom and pad_left == pad_right:
            padding = (pad_top, pad_left)
            pad2d = None
        else:
            padding = (0, 0)
            pad2d = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))

        if len(operation.dilations) == 2:
            dilation: Union[int, Tuple[int, int]] = (
                operation.dilations[0],
                operation.dilations[1],
            )
        else:
            dilation = operation.dilations[0]

        module: nn.Module = nn.Conv2d(
            in_c,
            out_c,
            operation.kernel_shape,
            stride=operation.strides,
            padding=padding,
            dilation=dilation,
            groups=operation.group,
        )
        module.weight.data = torch.from_numpy(operation.w)
        module.bias.data = torch.from_numpy(operation.b)
        if pad2d:
            module = nn.Sequential(pad2d, module)

        op = PytorchOperation(opname, module, inputs=[x.name], outputs=[opname])

        return op

    def visit_Flatten(self, operation: operations.Flatten) -> PytorchOperation:
        opname = self._get_name(operation)

        inputs = []

        x = operation.x
        if isinstance(x, Operation):
            x = self.visit(x)
            inputs.append(x.name)

        module = Flatten(operation.axis)

        op = PytorchOperation(opname, module, inputs=inputs, outputs=[opname])

        return op

    def visit_Gemm(self, operation: operations.Gemm) -> PytorchOperation:
        opname = self._get_name(operation)

        a = operation.a
        if isinstance(a, Operation):
            a = self.visit(a)
        b = operation.b
        if isinstance(b, Operation):
            b = self.visit(b)
        c = operation.c
        if isinstance(c, Operation):
            c = self.visit(c)

        assert operation.alpha == 1
        assert operation.beta == 1
        assert isinstance(c, np.ndarray)

        if isinstance(a, PytorchOperation) and isinstance(b, np.ndarray):
            assert len(b.shape) == 2
            assert not operation.transpose_a
            if operation.transpose_b:
                b = b.T
            module = nn.Linear(b.shape[0], b.shape[1])
            y = module(torch.randn(1, b.shape[0]))
            module.weight.data = torch.from_numpy(b.T)
            module.bias.data = torch.from_numpy(c)
            x = a
        elif isinstance(b, PytorchOperation) and isinstance(a, np.ndarray):
            assert len(a.shape) == 2
            assert not operation.transpose_b
            module = nn.Linear(a.shape[1], a.shape[0])
            module.weight.data = torch.from_numpy(a)
            module.bias.data = torch.from_numpy(c)
            x = a
        else:
            raise NotImplementedError()

        op = PytorchOperation(opname, module, inputs=[x.name], outputs=[opname])

        return op

    def visit_Input(self, operation: operations.Input) -> PytorchOperation:
        opname = self._get_name(operation)

        self.inputs.append(opname)

        return PytorchOperation(opname, None, [], [opname])

    def visit_MaxPool(self, operation: operations.MaxPool) -> PytorchOperation:
        opname = self._get_name(operation)

        x = operation.x
        if isinstance(x, Operation):
            x = self.visit(x)

        assert (
            operation.storage_order == operations.MaxPool.ROW_MAJOR_STORAGE
        ), "Only row major storage is currently supported for max pool operations"
        module: nn.Module = nn.MaxPool2d(
            tuple(operation.kernel_shape),
            stride=tuple(operation.strides),
            ceil_mode=operation.ceil_mode,
            dilation=tuple(operation.dilations),
        )
        if any(p != 0 for p in operation.pads):
            module = nn.Sequential(nn.ZeroPad2d(operation.pads), module)

        op = PytorchOperation(opname, module, inputs=[x.name], outputs=[opname])

        return op

    def visit_Relu(self, operation: operations.Relu) -> PytorchOperation:
        opname = self._get_name(operation)

        x = operation.x
        if isinstance(x, Operation):
            x = self.visit(x)

        op = PytorchOperation(opname, nn.ReLU(), inputs=[x.name], outputs=[opname])

        return op

    def visit_Reshape(self, operation: operations.Reshape) -> PytorchOperation:
        opname = self._get_name(operation)

        inputs = []

        x = operation.x
        if isinstance(x, Operation):
            x = self.visit(x)
            inputs.append(x.name)
        shape = operation.shape
        if isinstance(shape, Operation):
            shape = self.visit(shape)
            inputs.append(shape.name)

        module = View(shape)

        op = PytorchOperation(opname, module, inputs=inputs, outputs=[opname])

        return op

    def visit_Sigmoid(self, operation: operations.Sigmoid) -> PytorchOperation:
        opname = self._get_name(operation)

        x = operation.x
        if isinstance(x, Operation):
            x = self.visit(x)

        op = PytorchOperation(opname, nn.Sigmoid(), inputs=[x.name], outputs=[opname])

        return op

    def visit_Tanh(self, operation: operations.Tanh) -> PytorchOperation:
        opname = self._get_name(operation)

        x = operation.x
        if isinstance(x, Operation):
            x = self.visit(x)

        op = PytorchOperation(opname, nn.Tanh(), inputs=[x.name], outputs=[opname])

        return op

    def visit_Transpose(self, operation: operations.Transpose) -> PytorchOperation:
        opname = self._get_name(operation)

        inputs = []

        x = operation.x
        if isinstance(x, Operation):
            x = self.visit(x)
            inputs.append(x.name)
        permutation = operation.permutation
        if isinstance(permutation, Operation):
            permutation = self.visit(permutation)
            inputs.append(permutation.name)

        module = Permute(permutation)

        op = PytorchOperation(opname, module, inputs=inputs, outputs=[opname])

        return op
