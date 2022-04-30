from collections import namedtuple

import numpy as np
import onnx
import tensorflow as tf
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
