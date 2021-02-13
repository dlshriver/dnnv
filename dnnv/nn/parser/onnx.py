import onnx

from pathlib import Path
from typing import Set

from .. import OperationGraph
from ... import logging
from ..operations import Operation
from ..utils import as_numpy


def _parse_onnx_model(onnx_model: onnx.ModelProto) -> OperationGraph:
    logger = logging.getLogger(__name__)

    node_map = {}
    operation_map = {}
    parameter_map = {}
    for node in onnx_model.graph.node:
        if node.op_type in ["Constant"]:
            assert len(node.output) == 1
            parameter_map[node.output[0]] = as_numpy(node)
        else:
            for output_name in node.output:
                node_map[output_name] = node
    for initializer in onnx_model.graph.initializer:
        parameter_map[initializer.name] = as_numpy(initializer)
    for input_node in onnx_model.graph.input:
        if input_node.name not in parameter_map:
            operation_map[input_node.name] = Operation.from_onnx(input_node)

    operations = []
    visited = set()  # type: Set[int]

    def topo_sort(node):
        if id(node) in visited:
            return operation_map[id(node)]
        visited.add(id(node))
        inputs = []
        for name in node.input:
            if name in node_map:
                topo_sort(node_map[name])
                inputs.append(operation_map[name])
            elif name in parameter_map:
                inputs.append(parameter_map[name])
            elif name in operation_map:
                inputs.append(operation_map[name])
            else:
                raise ValueError("Unknown input name: %s" % name)
        operation = Operation.from_onnx(node, *inputs)
        if len(node.output) > 1:
            for i, output_name in enumerate(node.output):
                operation_map[output_name] = operation[i]
        else:
            operation_map[node.output[0]] = operation
        operation_map[id(node)] = operation
        operations.append(operation)

    for node in node_map.values():
        topo_sort(node)

    for i, operation in enumerate(operations, 1):
        logger.debug("%3d: %s", i, operation)

    output_operations = []
    for output_info in onnx_model.graph.output:
        output_operations.append(operation_map[output_info.name])

    return OperationGraph(output_operations)


def parse(path: Path) -> OperationGraph:
    onnx_model = onnx.load(str(path))
    return _parse_onnx_model(onnx_model)
