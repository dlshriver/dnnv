import numpy as np
import onnx
import tensorflow.compat.v1 as tf

from collections import namedtuple
from onnx import numpy_helper

TensorDetails = namedtuple("TensorDetails", ["shape", "dtype"])

ONNX_TO_NUMPY_DTYPE = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE

NUMPY_TO_ONNX_DTYPE = {v: k for k, v in ONNX_TO_NUMPY_DTYPE.items()}

ONNX_TO_TENSORFLOW_DTYPE = {
    onnx.TensorProto.DOUBLE: tf.float64,
    onnx.TensorProto.FLOAT16: tf.float16,
    onnx.TensorProto.FLOAT: tf.float32,
    onnx.TensorProto.INT8: tf.int8,
    onnx.TensorProto.INT16: tf.int16,
    onnx.TensorProto.INT32: tf.int32,
    onnx.TensorProto.INT64: tf.int64,
    onnx.TensorProto.UINT8: tf.uint8,
    onnx.TensorProto.UINT16: tf.uint16,
    onnx.TensorProto.UINT32: tf.uint32,
    onnx.TensorProto.UINT64: tf.uint64,
    onnx.TensorProto.BOOL: tf.bool,
    onnx.TensorProto.COMPLEX64: tf.complex64,
    onnx.TensorProto.COMPLEX128: tf.complex128,
    onnx.TensorProto.STRING: tf.string,
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
