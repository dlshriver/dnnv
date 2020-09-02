import numpy as np
import onnx

from collections import defaultdict
from typing import Any, Dict, List, Union

from .. import operations
from ..graph import OperationGraph
from ..operations import Operation
from ..utils import NUMPY_TO_ONNX_DTYPE
from ..visitors import OperationVisitor


def convert(op_graph: OperationGraph):
    converter = OnnxConverter(op_graph)
    model = converter.convert()

    return model


class OnnxConverter(OperationVisitor):
    def __init__(self, op_graph: OperationGraph):
        self.op_graph = op_graph
        self.inputs: List[onnx.ValueInfoProto] = []
        self.outputs: List[onnx.ValueInfoProto] = []
        self.initializer: List[onnx.TensorProto] = []
        self.visited: Dict[Operation, onnx.NodeProto] = {}
        self.op_counts: Dict[str, int] = defaultdict(int)

    def convert(self, name="onnx_model") -> onnx.ModelProto:
        output_details = self.op_graph.output_details
        for op, (shape, dtype) in zip(self.op_graph.output_operations, output_details):
            output_op = self.visit(op)
            node = onnx.helper.make_tensor_value_info(
                output_op.name, NUMPY_TO_ONNX_DTYPE[dtype], shape
            )
            self.outputs.append(node)

        nodes = [n for n in self.visited.values() if isinstance(n, onnx.NodeProto)]
        graph_def = onnx.helper.make_graph(
            nodes, name, self.inputs, self.outputs, initializer=self.initializer,
        )
        model_def = onnx.helper.make_model(graph_def, producer_name="dnnv")
        model_def = onnx.shape_inference.infer_shapes(model_def)
        onnx.checker.check_model(model_def)
        return model_def

    def visit(self, operation: Operation) -> Union[onnx.NodeProto, onnx.ValueInfoProto]:
        if operation not in self.visited:
            result = super().visit(operation)
            self.visited[operation] = result
        return self.visited[operation]

    def generic_visit(self, operation: Operation):
        if not hasattr(self, "visit_%s" % operation.__class__.__name__):
            raise ValueError(
                "ONNX converter not implemented for operation type %s"
                % operation.__class__.__name__
            )
        return super().generic_visit(operation)

    def _to_onnx_proto(
        self, value: Any, opname: str
    ) -> Union[onnx.NodeProto, onnx.TensorProto, onnx.ValueInfoProto]:
        if isinstance(value, Operation):
            return self.visit(value)
        elif isinstance(value, np.ndarray):
            tensor_proto = onnx.numpy_helper.from_array(value, name=opname)
            self.initializer.append(tensor_proto)
            return tensor_proto
        raise ValueError(f"Unknown type for operand of {opname}: {type(value)}")

    def visit_Gemm(self, operation: operations.Gemm) -> onnx.NodeProto:
        idx = self.op_counts["Gemm"] = self.op_counts["Gemm"] + 1
        opname = f"Gemm_{idx}"

        a = self._to_onnx_proto(operation.a, f"{opname}.a")
        b = self._to_onnx_proto(operation.b, f"{opname}.b")
        c = self._to_onnx_proto(operation.c, f"{opname}.c")

        node = onnx.helper.make_node(
            "Gemm",
            inputs=[a.name, b.name, c.name],
            outputs=[opname],
            alpha=operation.alpha,
            beta=operation.beta,
            transA=operation.transpose_a,
            transB=operation.transpose_b,
            name=opname,
        )

        return node

    def visit_Input(self, operation: operations.Input) -> onnx.ValueInfoProto:
        idx = self.op_counts["Input"] = self.op_counts["Input"] + 1
        opname = f"Input_{idx}"

        shape = operation.shape.tolist()
        dtype = NUMPY_TO_ONNX_DTYPE[operation.dtype]

        node = onnx.helper.make_tensor_value_info(opname, dtype, shape)
        self.inputs.append(node)

        return node

    def visit_Relu(self, operation: operations.Relu) -> onnx.NodeProto:
        idx = self.op_counts["Relu"] = self.op_counts["Relu"] + 1
        opname = f"Relu_{idx}"

        x = self._to_onnx_proto(operation.x, "Relu")

        node = onnx.helper.make_node(
            "Relu", inputs=[x.name], outputs=[opname], name=opname
        )

        return node
