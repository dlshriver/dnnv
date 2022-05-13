from collections import namedtuple

import numpy as np
import onnx
from onnx import numpy_helper

TensorDetails = namedtuple("TensorDetails", ["shape", "dtype"])

ONNX_TO_NUMPY_DTYPE = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE

NUMPY_TO_ONNX_DTYPE = {v: k for k, v in ONNX_TO_NUMPY_DTYPE.items()}


def as_numpy(node):
    if isinstance(node, onnx.TensorProto):
        return numpy_helper.to_array(node)
    if isinstance(node, onnx.NodeProto):
        return numpy_helper.to_array(node.attribute[0].t)
    if isinstance(node, onnx.AttributeProto):
        if node.type == onnx.AttributeProto.FLOAT:
            return float(node.f)
        if node.type == onnx.AttributeProto.INT:
            return int(node.i)
        if node.type == onnx.AttributeProto.INTS:
            return np.asarray(node.ints)
        if node.type == onnx.AttributeProto.STRING:
            return node.s.decode("utf-8")
        raise ValueError(f"Unknown attribute type: {node}")
    raise ValueError(f"Unknown node type: {type(node)}")
