from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import onnx

from .. import operations
from ..graph import OperationGraph
from ..operations import Operation
from ..utils import NUMPY_TO_ONNX_DTYPE
from ..visitors import OperationVisitor


def convert(op_graph: OperationGraph, *, add_missing_optional_inputs=False):
    converter = OnnxConverter(
        op_graph, add_missing_optional_inputs=add_missing_optional_inputs
    )
    model = converter.convert()

    return model


class OnnxConverter(OperationVisitor):
    def __init__(self, op_graph: OperationGraph, add_missing_optional_inputs=False):
        self.op_graph = op_graph
        self.inputs: List[onnx.ValueInfoProto] = []
        self.outputs: List[onnx.ValueInfoProto] = []
        self.initializer: List[onnx.TensorProto] = []
        self.visited: Dict[Operation, onnx.NodeProto] = {}
        self.op_counts: Dict[str, int] = defaultdict(int)
        self.add_missing_optional_inputs = add_missing_optional_inputs

    def convert(self, name="onnx_model") -> onnx.ModelProto:
        output_details = (
            self.op_graph.output_details
        )  # TODO: don't rely on tensorflow converter
        for op, (shape, dtype) in zip(self.op_graph.output_operations, output_details):
            output_op = self.visit(op)
            node = onnx.helper.make_tensor_value_info(
                output_op.name, NUMPY_TO_ONNX_DTYPE[dtype], shape
            )
            self.outputs.append(node)

        nodes = [n for n in self.visited.values() if isinstance(n, onnx.NodeProto)]
        graph_def = onnx.helper.make_graph(
            nodes,
            name,
            self.inputs,
            self.outputs,
            initializer=self.initializer,
        )
        # TODO : make opset configurable
        model_def = onnx.helper.make_model(
            graph_def,
            producer_name="dnnv",
            ir_version=7,
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )
        model_def = onnx.shape_inference.infer_shapes(model_def)
        onnx.checker.check_model(model_def, full_check=True)
        return model_def

    def visit(self, operation: Operation) -> Union[onnx.NodeProto, onnx.ValueInfoProto]:
        if operation not in self.visited:
            result = super().visit(operation)
            self.visited[operation] = result
        return self.visited[operation]

    def generic_visit(self, operation: Operation):
        if not hasattr(self, f"visit_{type(operation).__name__}"):
            raise ValueError(
                "ONNX converter not implemented"
                f" for operation type {type(operation).__name__}"
            )
        return super().generic_visit(operation)

    def _to_onnx_proto(
        self, value: Any, opname: str
    ) -> Union[onnx.NodeProto, onnx.TensorProto, onnx.ValueInfoProto]:
        if isinstance(value, Operation):
            return self.visit(value)
        if isinstance(value, np.ndarray):
            tensor_proto = onnx.numpy_helper.from_array(value, name=opname)
            self.initializer.append(tensor_proto)
            return tensor_proto
        if isinstance(value, bool):
            tensor_proto = onnx.numpy_helper.from_array(np.asarray(value), name=opname)
            self.initializer.append(tensor_proto)
            return tensor_proto
        if isinstance(value, (int, float)):
            tensor_proto = onnx.numpy_helper.from_array(
                np.array(value, dtype=f"{type(value).__name__}32"), name=opname
            )
            self.initializer.append(tensor_proto)
            return tensor_proto
        raise ValueError(f"Unknown type for operand of {opname}: {type(value)}")

    def visit_Add(self, operation: operations.Add) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        a = self._to_onnx_proto(operation.a, f"{opname}.a")
        b = self._to_onnx_proto(operation.b, f"{opname}.b")

        node = onnx.helper.make_node(
            op_type,
            inputs=[a.name, b.name],
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_Atan(self, operation: operations.Atan) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type, inputs=[x.name], outputs=[opname], name=opname
        )

        return node

    def visit_AveragePool(self, operation: operations.AveragePool) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name],
            outputs=[opname],
            kernel_shape=list(operation.kernel_shape),
            ceil_mode=operation.ceil_mode,
            count_include_pad=operation.count_include_pad,
            strides=list(operation.strides),
            pads=list(operation.pads),
            name=opname,
        )

        return node

    def visit_BatchNormalization(
        self, operation: operations.BatchNormalization
    ) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        scale = self._to_onnx_proto(operation.scale, f"{opname}.scale")
        bias = self._to_onnx_proto(operation.bias, f"{opname}.bias")
        mean = self._to_onnx_proto(operation.mean, f"{opname}.mean")
        variance = self._to_onnx_proto(operation.variance, f"{opname}.variance")

        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name, scale.name, bias.name, mean.name, variance.name],
            outputs=[opname],
            epsilon=operation.epsilon,
            momentum=operation.momentum,
            name=opname,
        )

        return node

    def visit_Cast(self, operation: operations.Cast) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        to = operation.to

        node = onnx.helper.make_node(
            op_type, inputs=[x.name], outputs=[opname], to=to, name=opname
        )

        return node

    def visit_Concat(self, operation: operations.Concat) -> onnx.NodeProto:
        idx = self.op_counts["Concat"] = self.op_counts["Concat"] + 1
        opname = f"Concat_{idx}"

        inputs = [
            self._to_onnx_proto(x, f"{opname}.x{i}") for i, x in enumerate(operation.x)
        ]

        node = onnx.helper.make_node(
            "Concat",
            inputs=[x.name for x in inputs],
            outputs=[opname],
            axis=operation.axis,
            name=opname,
        )

        return node

    def visit_Conv(self, operation: operations.Conv) -> onnx.NodeProto:
        idx = self.op_counts["Conv"] = self.op_counts["Conv"] + 1
        opname = f"Conv_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        w = self._to_onnx_proto(operation.w, f"{opname}.w")
        inputs = [x.name, w.name]
        if operation.b is not None:
            b = self._to_onnx_proto(operation.b, f"{opname}.b")
            inputs.append(b.name)
        elif self.add_missing_optional_inputs:
            b_ = np.zeros(operation.w.shape[0], dtype=operation.w.dtype)
            b = self._to_onnx_proto(b_, f"{opname}.b")
            inputs.append(b.name)

        node = onnx.helper.make_node(
            "Conv",
            inputs=inputs,
            outputs=[opname],
            kernel_shape=list(operation.kernel_shape),
            strides=list(operation.strides),
            dilations=list(operation.dilations),
            group=operation.group,
            pads=list(operation.pads),
            name=opname,
        )

        return node

    def visit_ConvTranspose(
        self, operation: operations.ConvTranspose
    ) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        w = self._to_onnx_proto(operation.w, f"{opname}.w")
        inputs = [x.name, w.name]
        if operation.b is not None:
            b = self._to_onnx_proto(operation.b, f"{opname}.b")
            inputs.append(b.name)
        elif self.add_missing_optional_inputs:
            b_ = np.zeros(w.shape[1], dtype=w.dtype)
            b = self._to_onnx_proto(b_, f"{opname}.b")
            inputs.append(b.name)

        extra_attributes = {}
        if operation.output_shape is not None:
            extra_attributes["output_shape"] = list(operation.output_shape)
        node = onnx.helper.make_node(
            op_type,
            inputs=inputs,
            outputs=[opname],
            auto_pad=operation.auto_pad,
            dilations=list(operation.dilations),
            group=operation.group,
            kernel_shape=list(operation.kernel_shape),
            output_padding=list(operation.output_padding),
            pads=list(operation.pads),
            strides=list(operation.strides),
            name=opname,
            **extra_attributes,
        )

        return node

    def visit_Div(self, operation: operations.Div) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        a = self._to_onnx_proto(operation.a, f"{opname}.a")
        b = self._to_onnx_proto(operation.b, f"{opname}.b")

        node = onnx.helper.make_node(
            op_type,
            inputs=[a.name, b.name],
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_Dropout(self, operation: operations.Dropout) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        ratio = self._to_onnx_proto(operation.ratio, f"{opname}.ratio")
        training_mode = self._to_onnx_proto(
            operation.training_mode, f"{opname}.training_mode"
        )

        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name, ratio.name, training_mode.name],
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_Elu(self, operation: operations.Elu) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name],
            alpha=operation.alpha,
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_Expand(self, operation: operations.Expand) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        shape = self._to_onnx_proto(operation.shape, f"{opname}.shape")

        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name, shape.name],
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_Flatten(self, operation: operations.Flatten) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name],
            outputs=[opname],
            axis=operation.axis,
            name=opname,
        )

        return node

    def visit_Gather(self, operation: operations.Gather) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        indices = self._to_onnx_proto(operation.indices, f"{opname}.indices")

        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name, indices.name],
            outputs=[opname],
            axis=operation.axis,
            name=opname,
        )

        return node

    def visit_Gemm(self, operation: operations.Gemm) -> onnx.NodeProto:
        idx = self.op_counts["Gemm"] = self.op_counts["Gemm"] + 1
        opname = f"Gemm_{idx}"

        a = self._to_onnx_proto(operation.a, f"{opname}.a")
        b = self._to_onnx_proto(operation.b, f"{opname}.b")
        inputs = [a.name, b.name]
        if operation.c is not None:
            c = self._to_onnx_proto(operation.c, f"{opname}.c")
            inputs.append(c.name)
        elif self.add_missing_optional_inputs:
            output_details = OperationGraph([operation]).output_details[0]
            c_ = np.zeros(output_details.shape[1], dtype=output_details.dtype)
            c = self._to_onnx_proto(c_, f"{opname}.c")
            inputs.append(c.name)

        node = onnx.helper.make_node(
            "Gemm",
            inputs=inputs,
            outputs=[opname],
            alpha=operation.alpha,
            beta=operation.beta,
            transA=operation.transpose_a,
            transB=operation.transpose_b,
            name=opname,
        )

        return node

    def visit_GlobalAveragePool(
        self, operation: operations.GlobalAveragePool
    ) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name],
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_Input(self, operation: operations.Input) -> onnx.ValueInfoProto:
        idx = self.op_counts["Input"] = self.op_counts["Input"] + 1
        opname = f"Input_{idx}"

        shape = np.asarray(operation.shape).tolist()
        if shape[0] < 0:
            shape[0] = 1
        dtype = NUMPY_TO_ONNX_DTYPE[operation.dtype]

        node = onnx.helper.make_tensor_value_info(opname, dtype, shape)
        self.inputs.append(node)

        return node

    def visit_MatMul(self, operation: operations.MatMul) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        a = self._to_onnx_proto(operation.a, f"{opname}.a")
        b = self._to_onnx_proto(operation.b, f"{opname}.b")

        node = onnx.helper.make_node(
            op_type,
            inputs=[a.name, b.name],
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_MaxPool(self, operation: operations.MaxPool) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name],
            outputs=[opname],
            kernel_shape=list(operation.kernel_shape),
            ceil_mode=operation.ceil_mode,
            strides=list(operation.strides),
            dilations=list(operation.dilations),
            pads=list(operation.pads),
            storage_order=operation.storage_order,
            name=opname,
        )

        return node

    def visit_Mul(self, operation: operations.Mul) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        a = self._to_onnx_proto(operation.a, f"{opname}.a")
        b = self._to_onnx_proto(operation.b, f"{opname}.b")

        node = onnx.helper.make_node(
            op_type,
            inputs=[a.name, b.name],
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_OutputSelect(self, operation: operations.OutputSelect) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        op = self._to_onnx_proto(operation.operation, f"{opname}.operation")

        node = onnx.helper.make_node(
            "Identity",
            inputs=[op.output[operation.index]],
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_Relu(self, operation: operations.Relu) -> onnx.NodeProto:
        idx = self.op_counts["Relu"] = self.op_counts["Relu"] + 1
        opname = f"Relu_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            "Relu", inputs=[x.name], outputs=[opname], name=opname
        )

        return node

    def visit_Reshape(self, operation: operations.Reshape) -> onnx.NodeProto:
        idx = self.op_counts["Reshape"] = self.op_counts["Reshape"] + 1
        opname = f"Reshape_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        shape = self._to_onnx_proto(operation.shape, f"{opname}.shape")

        if operation.allowzero:
            # TODO : need to use newer onnx opset version
            raise ValueError("Reshape allowzero is not yet supported")

        node = onnx.helper.make_node(
            "Reshape",
            inputs=[x.name, shape.name],
            # allowzero=operation.allowzero,
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_Sigmoid(self, operation: operations.Sigmoid) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type, inputs=[x.name], outputs=[opname], name=opname
        )

        return node

    def visit_Split(self, operation: operations.Split) -> onnx.NodeProto:
        op_type = str(operation)
        # TODO: split attribute is optional. Edits to nn/parser/onnx.py required.
        assert operation.split is not None
        idx = self.op_counts["Split"] = self.op_counts["Split"] + 1
        opname = f"Split_{idx}"
        outputs = []
        for i in range(len(operation.split)):
            outputs.append(f"output_{i}")
        outputs = np.array(outputs)
        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        split = self._to_onnx_proto(operation.split, f"{opname}.split")
        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name, split.name],
            outputs=outputs,
            name=opname,
            axis=operation.axis,
        )

        return node

    def visit_Slice(self, operation: operations.Slice) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        starts = self._to_onnx_proto(operation.starts, f"{opname}.starts")
        ends = self._to_onnx_proto(operation.ends, f"{opname}.ends")

        inputs = [x.name, starts.name, ends.name]
        if operation.steps is not None:
            axes = self._to_onnx_proto(operation.axes, f"{opname}.axes")
            steps = self._to_onnx_proto(operation.steps, f"{opname}.steps")
            inputs.extend([axes.name, steps.name])
        elif operation.axes is not None:
            axes = self._to_onnx_proto(operation.axes, f"{opname}.axes")
            inputs.append(axes.name)

        node = onnx.helper.make_node(
            op_type,
            inputs=inputs,
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_Sub(self, operation: operations.Sub) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        a = self._to_onnx_proto(operation.a, f"{opname}.a")
        b = self._to_onnx_proto(operation.b, f"{opname}.b")

        node = onnx.helper.make_node(
            op_type,
            inputs=[a.name, b.name],
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_Tanh(self, operation: operations.Tanh) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type, inputs=[x.name], outputs=[opname], name=opname
        )

        return node

    def visit_Transpose(self, operation: operations.Transpose) -> onnx.NodeProto:
        idx = self.op_counts["Transpose"] = self.op_counts["Transpose"] + 1
        opname = f"Transpose_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            "Transpose",
            inputs=[x.name],
            outputs=[opname],
            name=opname,
            perm=list(operation.permutation),
        )

        return node
