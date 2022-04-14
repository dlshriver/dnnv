import numpy as np
import onnx

from collections import namedtuple
from onnx import numpy_helper


TensorDetails = namedtuple("TensorDetails", ["shape", "dtype"])

ONNX_TO_NUMPY_DTYPE = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE

NUMPY_TO_ONNX_DTYPE = {v: k for k, v in ONNX_TO_NUMPY_DTYPE.items()}


def as_numpy(node):
    if isinstance(node, onnx.TensorProto):
        return numpy_helper.to_array(node)
    elif isinstance(node, onnx.NodeProto):
        return numpy_helper.to_array(node.attribute[0].t)
    elif isinstance(node, onnx.AttributeProto):
        if node.type == onnx.AttributeProto.FLOAT:
            return float(node.f)
        elif node.type == onnx.AttributeProto.INT:
            return int(node.i)
        elif node.type == onnx.AttributeProto.INTS:
            return np.asarray(node.ints)
        elif node.type == onnx.AttributeProto.STRING:
            return node.s.decode("utf-8")
        raise ValueError("Unknown attribute type: %s" % (node,))
    else:
        raise ValueError("Unknown node type: %s" % type(node))
