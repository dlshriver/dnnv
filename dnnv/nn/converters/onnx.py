from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import onnx
import onnx.numpy_helper

from .. import operations
from ..graph import OperationGraph
from ..operations import Operation
from ..utils import NUMPY_TO_ONNX_DTYPE
from .base import OperationConverter


class OnnxConverter(OperationConverter):
    def __init__(
        self,
        *,
        add_missing_optional_inputs=False,
        no_negative_input_dimensions=False,
        name="onnx_model",
        ir_version=7,
        opset=13,
    ):
        super().__init__()
        self.inputs: List[onnx.ValueInfoProto] = []
        self.initializer: List[onnx.TensorProto] = []
        self.visited: Dict[Operation, onnx.NodeProto] = self._cache
        self.op_counts: Dict[str, int] = defaultdict(int)
        self.add_missing_optional_inputs = add_missing_optional_inputs
        self.no_negative_input_dimensions = no_negative_input_dimensions
        self.ir_version = ir_version
        self.opset = opset

    @classmethod
    def convert(
        cls,
        op_graph: OperationGraph,
        *,
        add_missing_optional_inputs=False,
        name="onnx_model",
        ir_version=7,
        opset=13,
        no_negative_input_dimensions=False,
    ) -> onnx.ModelProto:
        self = cls(
            add_missing_optional_inputs=add_missing_optional_inputs,
            no_negative_input_dimensions=no_negative_input_dimensions,
            name=name,
            ir_version=ir_version,
            opset=opset,
        )

        output_names = {}
        for op in op_graph.outputs:
            output_op = self.visit(op)
            output_names[output_op.name] = op

        # TODO : need a more complete shape inference than is provided by onnx
        # try:
        #     outputs = []
        #     for op in op_graph.outputs:
        #         shape, dtype = op_graph.tensor_details[op]
        #         outputs.append(
        #             onnx.helper.make_tensor_value_info(
        #                 output_op.name, NUMPY_TO_ONNX_DTYPE[dtype], shape
        #             )
        #         )
        # except:
        #     outputs = []
        outputs = []

        nodes = [n for n in self.visited.values() if isinstance(n, onnx.NodeProto)]
        graph_def = onnx.helper.make_graph(
            nodes,
            name,
            self.inputs,
            outputs,
            initializer=self.initializer,
        )
        model_def = onnx.helper.make_model(
            graph_def,
            producer_name="dnnv",
            ir_version=ir_version,
            opset_imports=[onnx.helper.make_opsetid("", opset)],
        )
        if len(outputs) == 0:
            model_def = onnx.shape_inference.infer_shapes(model_def, data_prop=True)
            value_info_map = {
                value_info.name: value_info for value_info in model_def.graph.value_info
            }
            for value_info in model_def.graph.input:
                value_info_map[value_info.name] = value_info
            output_value_info = [value_info_map[name] for name in output_names]
            # output_value_info = []
            # for name in output_names:
            #     if hasattr(value_info_map[name].type, "shape"):
            #         output_value_info.append(value_info_map[name])
            #     else:
            #         shape, dtype = op_graph.tensor_details[output_names[name]]
            #         output_value_info.append(
            #             onnx.helper.make_tensor_value_info(
            #                 name, NUMPY_TO_ONNX_DTYPE[dtype], shape
            #             )
            #         )
            model_def.graph.output.extend(output_value_info)
        onnx.checker.check_model(model_def, full_check=True)
        return model_def

    def _to_onnx_proto(
        self, value: Any, opname: str
    ) -> Union[onnx.NodeProto, onnx.TensorProto, onnx.ValueInfoProto]:
        if isinstance(value, Operation):
            return self.visit(value)
        elif isinstance(value, (np.ndarray, np.number)):
            tensor_proto = onnx.numpy_helper.from_array(value, name=opname)
            self.initializer.append(tensor_proto)
            return tensor_proto
        elif isinstance(value, (int, float)):
            tensor_proto = onnx.numpy_helper.from_array(np.array(value), name=opname)
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
            b_ = np.zeros(w.shape[0], dtype=w.dtype)
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

        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name, ratio.name],
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

    def visit_Identity(self, operation: operations.Identity) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type, inputs=[x.name], outputs=[opname], name=opname
        )

        return node

    def visit_Input(self, operation: operations.Input) -> onnx.ValueInfoProto:
        idx = self.op_counts["Input"] = self.op_counts["Input"] + 1
        opname = f"Input_{idx}"

        shape = np.asarray(operation.shape).tolist()
        if self.no_negative_input_dimensions:
            shape = [d if d >= 0 else 1 for d in shape]
        dtype = NUMPY_TO_ONNX_DTYPE[np.dtype(operation.dtype)]

        node = onnx.helper.make_tensor_value_info(opname, dtype, shape)
        self.inputs.append(node)

        return node

    def visit_LeakyRelu(self, operation: operations.LeakyRelu) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name],
            outputs=[opname],
            name=opname,
            alpha=operation.alpha,
        )

        return node

    def visit_LogSoftmax(self, operation: operations.LogSoftmax) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type, inputs=[x.name], outputs=[opname], name=opname, axis=operation.axis
        )

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

        if operation.index != 0:
            raise NotImplementedError(
                "Support for operations with multiple ouputs is not yet implemented."
            )

        op = self._to_onnx_proto(operation.operation, f"{opname}.operation")

        node = onnx.helper.make_node(
            "Identity",
            inputs=[op.name],
            outputs=[opname],
            name=opname,
        )

        return node

    def visit_Pad(self, operation: operations.Pad) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        mode = operation.mode
        pads = self._to_onnx_proto(operation.pads, f"{opname}.pads")

        if mode == "constant" and operation.value:
            value = self._to_onnx_proto(operation.value, f"{opname}.value")
            node = onnx.helper.make_node(
                op_type,
                inputs=[x.name, pads.name, value.name],
                outputs=[opname],
                name=opname,
                mode=mode,
            )
        else:
            node = onnx.helper.make_node(
                op_type,
                inputs=[x.name, pads.name],
                outputs=[opname],
                name=opname,
                mode=mode,
            )

        return node

    def visit_RandomUniform(
        self, operation: operations.RandomUniform
    ) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        node = onnx.helper.make_node(
            op_type,
            inputs=[],
            outputs=[opname],
            name=opname,
            shape=operation.shape,
            dtype=NUMPY_TO_ONNX_DTYPE[np.dtype(operation.dtype)],
            high=operation.high,
            low=operation.low,
            seed=operation.seed,
        )

        return node

    def visit_Relu(self, operation: operations.Relu) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type, inputs=[x.name], outputs=[opname], name=opname
        )

        return node

    def visit_Reshape(self, operation: operations.Reshape) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        shape = self._to_onnx_proto(operation.shape, f"{opname}.shape")

        if self.opset < 13 and operation.allowzero:
            raise ValueError(
                f"{op_type} allowzero is not supported below ONNX opset 14"
            )

        if self.opset >= 14:
            node = onnx.helper.make_node(
                op_type,
                inputs=[x.name, shape.name],
                allowzero=operation.allowzero,
                outputs=[opname],
                name=opname,
            )
        else:
            node = onnx.helper.make_node(
                op_type,
                inputs=[x.name, shape.name],
                outputs=[opname],
                name=opname,
            )

        return node

    def visit_Resize(self, operation: operations.Resize) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        attributes = {}
        roi_name = ""
        scales_name = ""
        sizes_name = ""
        if operation.coordinate_transformation_mode == "tf_crop_and_resize":
            roi = self._to_onnx_proto(operation.roi, f"{opname}.roi")
            roi_name = roi.name
            if operation.extrapolation_value:
                attributes["extrapolation_value"] = operation.extrapolation_value
        if operation.scales is not None and operation.scales.size > 0:
            assert operation.sizes is None or operation.sizes.size == 0
            scales = self._to_onnx_proto(operation.scales, f"{opname}.scales")
            scales_name = scales.name
        else:
            sizes = self._to_onnx_proto(operation.sizes, f"{opname}.sizes")
            sizes_name = sizes.name

        if operation.mode == "cubic":
            attributes["cubic_coeff_a"] = operation.cubic_coeff_a
        elif operation.mode == "nearest":
            attributes["nearest_mode"] = operation.nearest_mode

        node = onnx.helper.make_node(
            op_type,
            inputs=[x.name, roi_name, scales_name, sizes_name],
            outputs=[opname],
            name=opname,
            coordinate_transformation_mode=operation.coordinate_transformation_mode,
            mode=operation.mode,
            exclude_outside=operation.exclude_outside,
            **attributes,
        )

        return node

    def visit_Shape(self, operation: operations.Shape) -> onnx.NodeProto:
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

    def visit_Sigmoid(self, operation: operations.Sigmoid) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type, inputs=[x.name], outputs=[opname], name=opname
        )

        return node

    def visit_Sign(self, operation: operations.Sign) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type, inputs=[x.name], outputs=[opname], name=opname
        )

        return node

    def visit_Softmax(self, operation: operations.Softmax) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")

        node = onnx.helper.make_node(
            op_type, inputs=[x.name], outputs=[opname], name=opname, axis=operation.axis
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

    def visit_Tile(self, operation: operations.Tile) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        repeats = self._to_onnx_proto(operation.repeats, f"{opname}.repeats")

        node = onnx.helper.make_node(
            op_type, inputs=[x.name, repeats.name], outputs=[opname], name=opname
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

    def visit_Unsqueeze(self, operation: operations.Unsqueeze) -> onnx.NodeProto:
        op_type = str(operation)
        idx = self.op_counts[op_type] = self.op_counts[op_type] + 1
        opname = f"{op_type}_{idx}"

        x = self._to_onnx_proto(operation.x, f"{opname}.x")
        axes = self._to_onnx_proto(operation.axes, f"{opname}.axes")

        node = onnx.helper.make_node(
            op_type, inputs=[x.name, axes.name], outputs=[opname], name=opname
        )

        return node


__all__ = ["OnnxConverter"]
