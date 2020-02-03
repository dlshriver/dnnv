import numpy as np
import onnx

from onnx import numpy_helper

ONNX_TO_NUMPY_DTYPE = {
    onnx.TensorProto.DOUBLE: np.dtype("float64"),
    onnx.TensorProto.FLOAT16: np.dtype("float16"),
    onnx.TensorProto.FLOAT: np.dtype("float32"),
    onnx.TensorProto.INT16: np.dtype("int16"),
    onnx.TensorProto.INT32: np.dtype("int32"),
    onnx.TensorProto.INT64: np.dtype("int64"),
}


def as_numpy(node):
    if isinstance(node, onnx.TensorProto):
        return numpy_helper.to_array(node)
    elif isinstance(node, onnx.NodeProto):
        return numpy_helper.to_array(node.attribute[0].t)
    elif isinstance(node, onnx.AttributeProto):
        if node.type == onnx.AttributeProto.FLOAT:
            return np.float(node.f)
        elif node.type == onnx.AttributeProto.INT:
            return np.int(node.i)
        elif node.type == onnx.AttributeProto.INTS:
            return np.asarray(node.ints)
        elif node.type == onnx.AttributeProto.STRING:
            return node.s.decode("utf-8")
        raise ValueError("Unknown attribute type: %s" % (node,))
    else:
        raise ValueError("Unknown node type: %s" % type(node))
