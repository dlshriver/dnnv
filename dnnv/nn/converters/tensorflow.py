import numpy as np
import tensorflow.compat.v1 as tf

from ..graph import OperationGraph
from ..operations import Operation
from ..utils import ONNX_TO_TENSORFLOW_DTYPE
from ..visitors import OperationVisitor


def convert(op_graph: OperationGraph):
    converter = TensorflowConverter()
    output_funcs = []
    for op in op_graph.output_operations:
        output_funcs.append(converter.visit(op))

    def func(*inputs, squeeze=True):
        concrete = True
        if any(isinstance(x, tf.Tensor) for x in inputs):
            concrete = False
        converter._cache.clear()
        graph = tf.Graph()
        with graph.as_default():
            outputs = []
            for output_func in output_funcs:
                output = output_func(*inputs)
                if isinstance(output, tf.Tensor) and tf.executing_eagerly():
                    output = output.numpy()
                outputs.append(output)
        if concrete:
            tensor_indices = []
            for i, output in enumerate(outputs):
                if isinstance(output, tf.Tensor):
                    tensor_indices.append(i)
            if len(tensor_indices) > 0:
                with tf.Session(graph=graph) as sess:
                    concrete_outputs = sess.run([outputs[i] for i in tensor_indices])
                for i, j in enumerate(tensor_indices):
                    outputs[j] = concrete_outputs[i]
        if squeeze and len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    return func


def _concretize(variables, inputs):
    concrete_values = []
    for variable in variables:
        if callable(variable):
            concrete_values.append(variable(*inputs))
        else:
            concrete_values.append(variable)
    if len(concrete_values) == 1:
        return concrete_values[0]
    return concrete_values


class TensorflowConverter(OperationVisitor):
    def __init__(self):
        self.input_count = 0
        self.results = {}
        self._cache = {}

    def _cached(self, func):
        def wrapped_func(*args, **kwargs):
            if func not in self._cache:
                self._cache[func] = func(*args, **kwargs)
            return self._cache[func]

        return wrapped_func

    def visit(self, operation):
        if operation not in self.results:
            result = super().visit(operation)
            self.results[operation] = result
        return self.results[operation]

    def generic_visit(self, operation):
        if not hasattr(self, "visit_%s" % operation.__class__.__name__):
            raise ValueError(
                "Tensorflow converter not implemented for operation type %s"
                % operation.__class__.__name__
            )
        return super().generic_visit(operation)

    def visit_Add(self, operation):
        a_ = operation.a
        if isinstance(a_, Operation):
            a_ = self.visit(a_)
        b_ = operation.b
        if isinstance(b_, Operation):
            b_ = self.visit(b_)

        @self._cached
        def add_func(*inputs):
            a, b = _concretize([a_, b_], inputs)
            result = tf.add(a, b)
            return result

        return add_func

    def visit_Atan(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def atan_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.atan(x)
            return result

        return atan_func

    def visit_AveragePool(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def avgpool_func(*inputs):
            x = _concretize([x_], inputs)
            kernel_shape = operation.kernel_shape
            strides = operation.strides
            pad_top, pad_left, pad_bottom, pad_right = operation.pads

            x = tf.transpose(x, (0, 2, 3, 1))
            padded_x = tf.pad(
                x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            )
            result = tf.nn.pool(
                padded_x,
                kernel_shape,
                pooling_type="AVG",
                strides=strides,
                padding="VALID",
            )
            result = tf.transpose(result, (0, 3, 1, 2))
            return result

        return avgpool_func

    def visit_BatchNormalization(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def batchnorm_func(*inputs):
            x = _concretize([x_], inputs)
            scale = operation.scale
            bias = operation.bias
            mean = operation.mean
            variance = operation.variance
            epsilon = operation.epsilon

            if len(x.shape) == 4:
                x = tf.transpose(x, (0, 2, 3, 1))
                result = tf.nn.batch_normalization(
                    x,
                    mean=mean,
                    variance=variance,
                    offset=bias,
                    scale=scale,
                    variance_epsilon=epsilon,
                )
                result = tf.transpose(result, (0, 3, 1, 2))
            else:
                result = tf.nn.batch_normalization(
                    x,
                    mean=mean,
                    variance=variance,
                    offset=bias,
                    scale=scale,
                    variance_epsilon=epsilon,
                )
            return result

        return batchnorm_func

    def visit_Cast(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def cast_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.cast(x, ONNX_TO_TENSORFLOW_DTYPE[operation.to])
            return result

        return cast_func

    def visit_Concat(self, operation):
        tensors_ = []
        for x in operation.x:
            if isinstance(x, Operation):
                x = self.visit(x)
            tensors_.append(x)

        @self._cached
        def concat_func(*inputs):
            tensors = x = _concretize(tensors_, inputs)
            result = tf.concat(tensors, axis=operation.axis)
            return result

        return concat_func

    def visit_Conv(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def conv_func(*inputs):
            x = _concretize([x_], inputs)
            weights = operation.w
            if operation.b is not None:
                bias = operation.b
            else:
                bias = np.zeros((weights.shape[0],), dtype=weights.dtype)
            assert np.all(operation.dilations == 1)
            assert np.all(operation.group == 1)
            pad_top, pad_left, pad_bottom, pad_right = operation.pads

            x = tf.transpose(x, (0, 2, 3, 1))
            padded_x = tf.pad(
                x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            )
            result = tf.nn.bias_add(
                tf.nn.conv2d(
                    padded_x,
                    weights.transpose((2, 3, 1, 0)),
                    operation.strides,
                    padding="VALID",
                ),
                bias,
            )
            result = tf.transpose(result, (0, 3, 1, 2))
            return result

        return conv_func

    def visit_Div(self, operation):
        a_ = operation.a
        if isinstance(a_, Operation):
            a_ = self.visit(a_)
        b_ = operation.b
        if isinstance(b_, Operation):
            b_ = self.visit(b_)

        @self._cached
        def div_func(*inputs):
            a, b = _concretize([a_, b_], inputs)
            result = tf.divide(a, b)
            return result

        return div_func

    def visit_Dropout(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def dropout_func(*inputs):
            x = _concretize([x_], inputs)
            return x, None

        return dropout_func

    def visit_Elu(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def elu_func(*inputs):
            if operation.alpha != 1.0:
                raise NotImplementedError(
                    "The tensorflow converter currently does not support ELU activations with alpha other than 1.0"
                )
            x = _concretize([x_], inputs)
            result = tf.nn.elu(x)
            return result

        return elu_func

    def visit_Expand(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def expand_func(*inputs):
            x = _concretize([x_], inputs)
            shape = operation.shape
            result = x * tf.ones(shape, x.dtype)
            return result

        return expand_func

    def visit_Flatten(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def flatten_func(*inputs):
            x = _concretize([x_], inputs)
            axis = operation.axis
            new_shape = (1, -1) if axis == 0 else (int(np.prod(x.shape[:axis])), -1)
            result = tf.reshape(x, new_shape)
            return result

        return flatten_func

    def visit_Gather(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)
        indices_ = operation.indices
        if isinstance(indices_, Operation):
            indices_ = self.visit(indices_)

        @self._cached
        def gather_func(*inputs):
            x, indices = _concretize([x_, indices_], inputs)
            result = tf.gather(x, indices, axis=operation.axis)
            return result

        return gather_func

    def visit_Gemm(self, operation):
        a_ = operation.a
        if isinstance(a_, Operation):
            a_ = self.visit(a_)
        b_ = operation.b
        if isinstance(b_, Operation):
            b_ = self.visit(b_)
        c_ = operation.c
        if isinstance(c_, Operation):
            c_ = self.visit(c_)

        @self._cached
        def gemm_func(*inputs):
            a, b, c = _concretize([a_, b_, c_], inputs)
            if isinstance(a, np.ndarray) and isinstance(b, tf.Tensor):
                a = a.astype(b.dtype.as_numpy_dtype)
            if isinstance(b, np.ndarray) and isinstance(a, tf.Tensor):
                b = b.astype(a.dtype.as_numpy_dtype)
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                a = a.astype(b.dtype)
            result = (
                operation.alpha
                * tf.matmul(
                    a,
                    b,
                    transpose_a=operation.transpose_a,
                    transpose_b=operation.transpose_b,
                )
                + operation.beta * c
            )
            return result

        return gemm_func

    def visit_GlobalAveragePool(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def globalavgpool_func(*inputs):
            x = _concretize([x_], inputs)

            x = tf.transpose(x, (0, 2, 3, 1))
            result = tf.nn.pool(
                x, x.shape[1:3], pooling_type="AVG", strides=(1, 1), padding="VALID"
            )
            result = tf.transpose(result, (0, 3, 1, 2))
            return result

        return globalavgpool_func

    def visit_Identity(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def identity_func(*inputs):
            x = _concretize([x_], inputs)
            return x

        return identity_func

    def visit_Input(self, operation):
        input_idx = self.input_count

        @self._cached
        def input_func(*inputs):
            x = inputs[input_idx]
            if any(
                d1 != d2 and -1 not in (d1, d2) and None not in (d1, d2)
                for d1, d2 in zip(operation.shape[1:], x.shape[1:])
            ):
                raise ValueError(
                    "Incorrect input shape: %s != %s" % (operation.shape, x.shape)
                )
            if x.dtype != operation.dtype:
                raise TypeError(
                    "Incorrect type, %s, for input %d. Expected type %s."
                    % (x.dtype, input_idx, operation.dtype)
                )
            return x

        self.input_count += 1

        return input_func

    def visit_MatMul(self, operation):
        a_ = operation.a
        if isinstance(a_, Operation):
            a_ = self.visit(a_)
        b_ = operation.b
        if isinstance(b_, Operation):
            b_ = self.visit(b_)

        @self._cached
        def matmul_func(*inputs):
            a, b = _concretize([a_, b_], inputs)
            if isinstance(a, np.ndarray) and isinstance(b, tf.Tensor):
                a = a.astype(b.dtype.as_numpy_dtype)
            if isinstance(b, np.ndarray) and isinstance(a, tf.Tensor):
                b = b.astype(a.dtype.as_numpy_dtype)
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                a = a.astype(b.dtype)

            if len(a.shape) == 2 and len(b.shape) == 1:
                result = tf.matmul(a, b[:, None])[:, 0]
            elif len(a.shape) == 1 and len(b.shape) == 2:
                result = tf.matmul(a[None], b)[0]
            else:
                result = tf.matmul(a, b)
            return result

        return matmul_func

    def visit_MaxPool(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def maxpool_func(*inputs):
            x = _concretize([x_], inputs)
            kernel_shape = operation.kernel_shape
            strides = operation.strides
            pad_top, pad_left, pad_bottom, pad_right = operation.pads

            x = tf.transpose(x, (0, 2, 3, 1))
            padded_x = tf.pad(
                x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            )
            result = tf.nn.pool(
                padded_x,
                kernel_shape,
                pooling_type="MAX",
                strides=strides,
                padding="VALID",
            )
            result = tf.transpose(result, (0, 3, 1, 2))
            return result

        return maxpool_func

    def visit_Mul(self, operation):
        a_ = operation.a
        if isinstance(a_, Operation):
            a_ = self.visit(a_)
        b_ = operation.b
        if isinstance(b_, Operation):
            b_ = self.visit(b_)

        @self._cached
        def mul_func(*inputs):
            a, b = _concretize([a_, b_], inputs)
            result = tf.multiply(a, b)
            return result

        return mul_func

    def visit_OutputSelect(self, operation):
        x_ = self.visit(operation.operation)

        @self._cached
        def output_select_func(*inputs):
            x = _concretize([x_], inputs)
            return x[operation.index]

        return output_select_func

    def visit_Pad(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def pad_func(*inputs):
            x = _concretize([x_], inputs)
            num_pads = len(operation.pads)
            pads = tuple(
                zip(operation.pads[: num_pads // 2], operation.pads[num_pads // 2 :])
            )
            result = tf.pad(
                x, pads, mode=operation.mode, constant_values=operation.value
            )
            return result

        return pad_func

    def visit_Relu(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def relu_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.nn.relu(x)
            return result

        return relu_func

    def visit_Reshape(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)
        shape_ = operation.shape
        if isinstance(shape_, Operation):
            shape_ = self.visit(shape_)

        @self._cached
        def reshape_func(*inputs):
            x, shape = _concretize([x_, shape_], inputs)
            result = tf.reshape(x, shape)
            assert result.shape[0] == x.shape[0]
            return result

        return reshape_func

    def visit_Shape(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def shape_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.shape(x)
            return result

        return shape_func

    def visit_Sigmoid(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def sigmoid_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.nn.sigmoid(x)
            return result

        return sigmoid_func

    def visit_Softmax(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def softmax_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.nn.softmax(x, axis=operation.axis)
            return result

        return softmax_func

    def visit_Sub(self, operation):
        a_ = operation.a
        if isinstance(a_, Operation):
            a_ = self.visit(a_)
        b_ = operation.b
        if isinstance(b_, Operation):
            b_ = self.visit(b_)

        @self._cached
        def sub_func(*inputs):
            a, b = _concretize([a_, b_], inputs)
            result = tf.subtract(a, b)
            return result

        return sub_func

    def visit_Tile(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def tile_func(*inputs):
            x = _concretize([x_], inputs)
            repeats = operation.repeats
            result = tf.tile(x, repeats)
            return result

        return tile_func

    def visit_Transpose(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def transpose_func(*inputs):
            x = _concretize([x_], inputs)
            permutation = operation.permutation
            result = tf.transpose(x, permutation)
            return result

        return transpose_func

    def visit_Unsqueeze(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def unsqueeze_func(*inputs):
            x = _concretize([x_], inputs)
            for axis in operation.axes:
                x = tf.expand_dims(x, axis)
            return x

        return unsqueeze_func

    def visit_Tanh(self, operation):
        x_ = operation.x
        if isinstance(x_, Operation):
            x_ = self.visit(x_)

        @self._cached
        def tanh_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.nn.tanh(x)
            return result

        return tanh_func
