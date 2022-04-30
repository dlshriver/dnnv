from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set

import onnx

from .. import OperationGraph
from ..operations import Operation
from ..utils import as_numpy


def traverse_nodes(
    node: onnx.NodeProto,
    node_map: Dict[str, onnx.NodeProto],
    operation_map: Dict[str, Operation],
    parameter_map: Dict[str, onnx.NodeProto],
    visited: Optional[Set[onnx.NodeProto]] = None,
):
    if visited is None:
        visited = set()
    if id(node) in visited:
        return
    visited.add(id(node))
    inputs = []
    for name in node.input:
        if name in node_map:
            traverse_nodes(
                node_map[name],
                node_map,
                operation_map,
                parameter_map,
                visited=visited,
            )
            inputs.append(operation_map[name])
        elif name in parameter_map:
            inputs.append(parameter_map[name])
        elif name in operation_map:
            inputs.append(operation_map[name])
        else:
            raise ValueError(f"Unknown input name: {name}")
    operation = Operation.from_onnx(node, *inputs)
    if len(node.output) > 1:
        for i, output_name in enumerate(node.output):
            operation_map[output_name] = operation[i]
    else:
        operation_map[node.output[0]] = operation


def _parse_onnx_model(onnx_model: onnx.ModelProto) -> OperationGraph:
    node_map: Dict[str, onnx.NodeProto] = {}
    operation_map: Dict[str, Operation] = {}
    parameter_map: Dict[str, Any] = {}
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

    for node in node_map.values():
        traverse_nodes(node, node_map, operation_map, parameter_map)

    output_operations = []
    for output_info in onnx_model.graph.output:
        output_operations.append(operation_map[output_info.name])

    return OperationGraph(output_operations)


def parse(path: Path) -> OperationGraph:
    onnx_model = onnx.load(str(path))
    return _parse_onnx_model(onnx_model)
