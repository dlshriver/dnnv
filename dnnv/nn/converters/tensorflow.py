import numpy as np
import tensorflow as tf

from .. import operations
from ..graph import OperationGraph
from ..operations import Operation
from ..utils import ONNX_TO_TENSORFLOW_DTYPE
from ..visitors import OperationVisitor


class TensorflowConverterError(Exception):
    pass


def convert(op_graph: OperationGraph):
    converter = TensorflowConverter()
    output_funcs = []
    for op in op_graph.output_operations:
        output_funcs.append(converter.visit(op))

    def func(*inputs, squeeze=True):
        converter.clear_cache()
        outputs = []
        for output_func in output_funcs:
            output = output_func(*inputs)
            if isinstance(output, tf.Tensor) and tf.executing_eagerly():
                output = output.numpy()
            outputs.append(output)
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
                try:
                    self._cache[func] = func(*args, **kwargs)
                except TensorflowConverterError:
                    raise
                except Exception as e:
                    args_str = ""
                    if len(e.args) == 1:
                        args_str = e.args[0]
                    elif len(e.args) > 1:
                        args_str = str(e.args)
                    raise TensorflowConverterError(
                        f"{type(e).__name__}: {args_str}"
                    ).with_traceback(e.__traceback__)
            return self._cache[func]

        return wrapped_func

    def clear_cache(self):
        self._cache.clear()

    def visit(self, operation):
        if operation is None or not isinstance(operation, Operation):
            return lambda *_, op=operation: op
        if operation not in self.results:
            result = super().visit(operation)
            self.results[operation] = result
        return self.results[operation]

    def generic_visit(self, operation):
        if not hasattr(self, f"visit_{type(operation).__name__}"):
            raise ValueError(
                "Tensorflow converter not implemented"
                f" for operation type {type(operation).__name__}"
            )
        return super().generic_visit(operation)

    def visit_Add(self, operation: operations.Add):
        a_ = self.visit(operation.a)
        b_ = self.visit(operation.b)

        @self._cached
        def add_func(*inputs):
            a, b = _concretize([a_, b_], inputs)
            result = tf.add(a, b)
            return result

        return add_func

    def visit_Atan(self, operation: operations.Atan):
        x_ = self.visit(operation.x)

        @self._cached
        def atan_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.atan(x)
            return result

        return atan_func

    def visit_AveragePool(self, operation: operations.AveragePool):
        x_ = self.visit(operation.x)

        @self._cached
        def avgpool_func(*inputs):
            x = _concretize([x_], inputs)
            if operation.ceil_mode:
                # TODO : add support
                raise ValueError(
                    "ceil_mode=True is not currently supported for AveragePool"
                )
            if any(p != 0 for p in operation.pads) and not operation.count_include_pad:
                # TODO : add support
                raise ValueError(
                    "count_include_pad=False is not currently supported for AveragePool"
                )
            kernel_shape = operation.kernel_shape
            strides = operation.strides
            num_pads = len(operation.pads)
            pads = tuple(
                zip(
                    operation.pads[: num_pads // 2],
                    operation.pads[num_pads // 2 :],
                )
            )

            x_ndim = int(tf.rank(x))
            x = tf.transpose(x, (0,) + tuple(range(2, x_ndim)) + (1,))
            padded_x = tf.pad(x, ((0, 0),) + pads + ((0, 0),))
            result = tf.nn.pool(
                padded_x,
                kernel_shape,
                pooling_type="AVG",
                strides=strides,
                padding="VALID",
            )
            result_ndim = int(tf.rank(result))
            result = tf.transpose(
                result, (0, result_ndim - 1) + tuple(range(1, result_ndim - 1))
            )
            return result

        return avgpool_func

    def visit_BatchNormalization(self, operation: operations.BatchNormalization):
        x_ = self.visit(operation.x)
        scale_ = self.visit(operation.scale)
        bias_ = self.visit(operation.bias)
        mean_ = self.visit(operation.mean)
        variance_ = self.visit(operation.variance)

        @self._cached
        def batchnorm_func(*inputs):
            x, scale, bias, mean, variance = _concretize(
                [x_, scale_, bias_, mean_, variance_], inputs
            )
            epsilon = operation.epsilon

            x_ndim = int(tf.rank(x))
            x = tf.transpose(x, (0,) + tuple(range(2, x_ndim)) + (1,))
            result = tf.nn.batch_normalization(
                x,
                mean=mean,
                variance=variance,
                offset=bias,
                scale=scale,
                variance_epsilon=epsilon,
            )
            result_ndim = int(tf.rank(result))
            result = tf.transpose(
                result, (0, result_ndim - 1) + tuple(range(1, result_ndim - 1))
            )
            return result

        return batchnorm_func

    def visit_Cast(self, operation: operations.Cast):
        x_ = self.visit(operation.x)

        @self._cached
        def cast_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.cast(x, ONNX_TO_TENSORFLOW_DTYPE[operation.to])
            return result

        return cast_func

    def visit_Concat(self, operation: operations.Concat):
        tensors_ = []
        for x in operation.x:
            tensors_.append(self.visit(x))

        @self._cached
        def concat_func(*inputs):
            tensors = _concretize(tensors_, inputs)
            result = tf.concat(tensors, axis=operation.axis)
            return result

        return concat_func

    def visit_Conv(self, operation: operations.Conv):
        x_ = self.visit(operation.x)
        w_ = self.visit(operation.w)
        b_ = self.visit(operation.b)

        @self._cached
        def conv_func(*inputs):
            x, weights, bias = _concretize([x_, w_, b_], inputs)
            if len(operation.kernel_shape) != 2:
                raise NotImplementedError(
                    "Non 2d convolutions are not currently supported."
                )
            if bias is None:
                bias = np.zeros((weights.shape[0],), dtype=weights.dtype)
            assert np.all(operation.dilations == 1)
            num_pads = len(operation.pads)
            pads = tuple(
                zip(
                    operation.pads[: num_pads // 2],
                    operation.pads[num_pads // 2 :],
                )
            )

            x_ndim = int(tf.rank(x))
            x = tf.transpose(x, (0,) + tuple(range(2, x_ndim)) + (1,))
            padded_x = tf.pad(x, ((0, 0),) + pads + ((0, 0),))
            result = tf.nn.bias_add(
                tf.nn.conv2d(
                    padded_x,
                    weights.transpose((2, 3, 1, 0)),
                    operation.strides,
                    padding="VALID",
                ),
                bias,
            )
            result_ndim = int(tf.rank(result))
            result = tf.transpose(
                result, (0, result_ndim - 1) + tuple(range(1, result_ndim - 1))
            )
            return result

        return conv_func

    def visit_ConvTranspose(self, operation: operations.ConvTranspose):
        x_ = self.visit(operation.x)
        w_ = self.visit(operation.w)
        b_ = self.visit(operation.b)

        @self._cached
        def convtranspose_func(*inputs):
            x, weights, bias = _concretize([x_, w_, b_], inputs)

            if len(operation.kernel_shape) == 1:
                conv_transpose = tf.nn.conv1d_transpose
            elif len(operation.kernel_shape) == 2:
                conv_transpose = tf.nn.conv2d_transpose
            elif len(operation.kernel_shape) == 3:
                conv_transpose = tf.nn.conv3d_transpose
            else:
                raise NotImplementedError(
                    f"{len(operation.kernel_shape)}d ConvTranspose"
                    " operations are not currently supported."
                )
            if (
                operation.auto_pad != "NOTSET"
                or operation.auto_pad == "VALID"
                and any(p != 0 for p in operation.pads)
            ):
                raise NotImplementedError(
                    f"Unsupported padding for ConvTranspose: {operation.auto_pad}"
                )
            if np.any(operation.dilations != 1):
                raise NotImplementedError(
                    f"Unsupported dilations for ConvTranspose: {operation.dilations}"
                )

            if bias is None:
                bias = np.zeros((weights.shape[1],), dtype=weights.dtype)

            num_pads = len(operation.pads)
            pads = tuple(
                zip(
                    operation.pads[: num_pads // 2],
                    operation.pads[num_pads // 2 :],
                )
            )
            if any(p != 0 for p in operation.pads):
                raise NotImplementedError(
                    "Non 0 pads are not currently supported for ConvTranspose"
                )

            output_shape = operation.output_shape
            if output_shape is None:
                input_shape = [int(d) for d in x.shape[2:]]
                start_pads = operation.pads[: num_pads // 2]
                end_pads = operation.pads[num_pads // 2 :]
                output_shape = (
                    [int(x.shape[0])]
                    + [
                        (
                            operation.strides[i] * (input_shape[i] - 1)
                            + operation.output_padding[i]
                            + (
                                (operation.kernel_shape[i] - 1) * operation.dilations[i]
                                + 1
                            )
                            - start_pads[i]
                            - end_pads[i]
                        )
                        for i in range(len(operation.kernel_shape))
                    ]
                    + [weights.shape[1]]
                )

            x_ndim = int(tf.rank(x))
            x = tf.transpose(x, (0,) + tuple(range(2, x_ndim)) + (1,))
            padded_x = tf.pad(x, ((0, 0),) + pads + ((0, 0),))
            weights_ndim = int(tf.rank(weights))
            result = tf.nn.bias_add(
                conv_transpose(
                    padded_x,
                    weights.transpose(tuple(range(2, weights_ndim)) + (1, 0)),
                    output_shape,
                    strides=operation.strides,
                    padding="VALID",
                    dilations=operation.dilations,
                ),
                bias,
            )
            result_ndim = int(tf.rank(result))
            result = tf.transpose(
                result, (0, result_ndim - 1) + tuple(range(1, result_ndim - 1))
            )
            return result

        return convtranspose_func

    def visit_Div(self, operation: operations.Div):
        a_ = self.visit(operation.a)
        b_ = self.visit(operation.b)

        @self._cached
        def div_func(*inputs):
            a, b = _concretize([a_, b_], inputs)
            result = tf.convert_to_tensor(tf.divide(a, b))
            return result

        return div_func

    def visit_Dropout(self, operation: operations.Dropout):
        x_ = self.visit(operation.x)

        @self._cached
        def dropout_func(*inputs):
            x = _concretize([x_], inputs)
            assert not operation.training_mode
            return x

        return dropout_func

    def visit_Elu(self, operation: operations.Elu):
        x_ = self.visit(operation.x)

        @self._cached
        def elu_func(*inputs):
            if operation.alpha != 1.0:
                raise NotImplementedError(
                    "Elu operations with alpha other than 1.0"
                    " are not currently supported."
                )
            x = _concretize([x_], inputs)
            result = tf.nn.elu(x)
            return result

        return elu_func

    def visit_Expand(self, operation: operations.Expand):
        x_ = self.visit(operation.x)
        shape_ = self.visit(operation.shape)

        @self._cached
        def expand_func(*inputs):
            x, shape = _concretize([x_, shape_], inputs)
            result = x * tf.ones(shape, x.dtype)
            return result

        return expand_func

    def visit_Flatten(self, operation: operations.Flatten):
        x_ = self.visit(operation.x)

        @self._cached
        def flatten_func(*inputs):
            x = _concretize([x_], inputs)
            axis = operation.axis
            new_shape = (1, -1) if axis == 0 else (int(np.prod(x.shape[:axis])), -1)
            result = tf.reshape(x, new_shape)
            return result

        return flatten_func

    def visit_Gather(self, operation: operations.Gather):
        x_ = self.visit(operation.x)
        indices_ = self.visit(operation.indices)

        @self._cached
        def gather_func(*inputs):
            x, indices = _concretize([x_, indices_], inputs)
            result = tf.gather(x, indices, axis=operation.axis)
            return result

        return gather_func

    def visit_Gemm(self, operation: operations.Gemm):
        a_ = self.visit(operation.a)
        b_ = self.visit(operation.b)
        c_ = self.visit(operation.c)

        @self._cached
        def gemm_func(*inputs):
            a, b, c = _concretize([a_, b_, c_], inputs)
            result = operation.alpha * tf.matmul(
                a,
                b,
                transpose_a=operation.transpose_a,
                transpose_b=operation.transpose_b,
            )
            if c is not None:
                result = result + operation.beta * c
            return result

        return gemm_func

    def visit_GlobalAveragePool(self, operation: operations.GlobalAveragePool):
        x_ = self.visit(operation.x)

        @self._cached
        def globalavgpool_func(*inputs):
            x = _concretize([x_], inputs)

            x = tf.transpose(x, (0, 2, 3, 1))
            result = tf.nn.pool(
                x,
                x.shape[1:3],
                pooling_type="AVG",
                strides=(1, 1),
                padding="VALID",
            )
            result = tf.transpose(result, (0, 3, 1, 2))
            return result

        return globalavgpool_func

    def visit_Identity(self, operation: operations.Identity):
        x_ = self.visit(operation.x)

        @self._cached
        def identity_func(*inputs):
            x = _concretize([x_], inputs)
            return tf.convert_to_tensor(x)

        return identity_func

    def visit_Input(self, operation: operations.Input):
        input_idx = self.input_count

        @self._cached
        def input_func(*inputs):
            x = inputs[input_idx]
            if any(
                d1 != d2 and -1 not in (d1, d2) and None not in (d1, d2)
                for d1, d2 in zip(operation.shape[1:], x.shape[1:])
            ):
                raise ValueError(
                    f"Incorrect input shape: {operation.shape} != {x.shape}"
                )
            if x.dtype != operation.dtype:
                raise TypeError(
                    f"Incorrect type, {x.dtype}, for input {input_idx}."
                    f" Expected type {operation.dtype}."
                )
            return tf.convert_to_tensor(x)

        self.input_count += 1

        return input_func

    def visit_LeakyRelu(self, operation: operations.LeakyRelu):
        x_ = self.visit(operation.x)

        @self._cached
        def leakyrelu_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.nn.leaky_relu(x, alpha=operation.alpha)
            return result

        return leakyrelu_func

    def visit_LogSoftmax(self, operation: operations.LogSoftmax):
        x_ = self.visit(operation.x)

        @self._cached
        def softmax_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.nn.log_softmax(x, axis=operation.axis)
            return result

        return softmax_func

    def visit_MatMul(self, operation: operations.MatMul):
        a_ = self.visit(operation.a)
        b_ = self.visit(operation.b)

        @self._cached
        def matmul_func(*inputs):
            a, b = _concretize([a_, b_], inputs)

            if len(a.shape) == 2 and len(b.shape) == 1:
                result = tf.matmul(a, b[:, None])[:, 0]
            elif len(a.shape) == 1 and len(b.shape) == 2:
                result = tf.matmul(a[None], b)[0]
            else:
                result = tf.matmul(a, b)
            return result

        return matmul_func

    def visit_MaxPool(self, operation: operations.MaxPool):
        x_ = self.visit(operation.x)

        @self._cached
        def maxpool_func(*inputs):
            x = _concretize([x_], inputs)

            if operation.ceil_mode:
                # TODO : add support
                raise ValueError(
                    "ceil_mode=True is not currently supported for MaxPool"
                )

            kernel_shape = operation.kernel_shape
            strides = operation.strides
            num_pads = len(operation.pads)
            pads = tuple(
                zip(
                    operation.pads[: num_pads // 2],
                    operation.pads[num_pads // 2 :],
                )
            )

            x_ndim = int(tf.rank(x))
            x = tf.transpose(x, (0,) + tuple(range(2, x_ndim)) + (1,))
            padded_x = tf.pad(
                x, ((0, 0),) + pads + ((0, 0),), constant_values=x.dtype.min
            )
            result = tf.nn.pool(
                padded_x,
                kernel_shape,
                pooling_type="MAX",
                strides=strides,
                dilations=operation.dilations,
                padding="VALID",
            )
            result_ndim = int(tf.rank(result))
            result = tf.transpose(
                result, (0, result_ndim - 1) + tuple(range(1, result_ndim - 1))
            )
            return result

        return maxpool_func

    def visit_Mul(self, operation: operations.Mul):
        a_ = self.visit(operation.a)
        b_ = self.visit(operation.b)

        @self._cached
        def mul_func(*inputs):
            a, b = _concretize([a_, b_], inputs)
            result = tf.multiply(a, b)
            return result

        return mul_func

    def visit_OutputSelect(self, operation: operations.OutputSelect):
        x_ = self.visit(operation.operation)

        @self._cached
        def output_select_func(*inputs):
            x = _concretize([x_], inputs)
            return x[operation.index]

        return output_select_func

    def visit_Pad(self, operation: operations.Pad):
        x_ = self.visit(operation.x)
        pads_ = self.visit(operation.pads)
        value_ = self.visit(operation.value)

        @self._cached
        def pad_func(*inputs):
            x, pads, value = _concretize([x_, pads_, value_], inputs)
            mode = operation.mode.upper()
            if mode != "CONSTANT":
                raise ValueError(f"{mode} padding is not currently supported")
            num_pads = len(pads)
            pads_tuple = tuple(
                zip(
                    pads[: num_pads // 2],
                    pads[num_pads // 2 :],
                )
            )
            result = tf.pad(x, pads_tuple, mode=mode, constant_values=value)
            return result

        return pad_func

    def visit_Relu(self, operation: operations.Relu):
        x_ = self.visit(operation.x)

        @self._cached
        def relu_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.nn.relu(x)
            return result

        return relu_func

    def visit_Reshape(self, operation: operations.Reshape):
        x_ = self.visit(operation.x)
        shape_ = self.visit(operation.shape)

        @self._cached
        def reshape_func(*inputs):
            x, shape = _concretize([x_, shape_], inputs)
            if not operation.allowzero:
                for i, d in enumerate(shape):
                    if d == 0:
                        shape[i] = x.shape[i]
            result = tf.reshape(x, shape)
            return result

        return reshape_func

    def visit_Resize(self, operation: operations.Resize):
        x_ = self.visit(operation.x)
        roi_ = self.visit(operation.roi)
        scales_ = self.visit(operation.scales)
        sizes_ = self.visit(operation.sizes)

        @self._cached
        def resize_func(*inputs):
            x, roi, scales, sizes = _concretize([x_, roi_, scales_, sizes_], inputs)
            assert operation.coordinate_transformation_mode in [
                "asymmetric",
                "tf_crop_and_resize",
            ]
            assert operation.mode in ["nearest", "linear"]
            assert operation.exclude_outside == 0
            if roi is None or roi.size == 0:
                roi = np.array([[0, 0, 1, 1]])
            else:
                assert operation.coordinate_transformation_mode == "tf_crop_and_resize"
                assert roi.size == 8 and roi.ndim == 1
                roi = roi[None, [2, 3, 6, 7]]
            if sizes is None or sizes.size == 0:
                assert scales[0] == 1.0 and scales[1] == 1.0
                sizes = (scales * [int(d) for d in x.shape]).astype(int)
            assert sizes.ndim == 1 and sizes.size == 4
            sizes = sizes[2:]
            method = operation.mode
            if method == "linear":
                method = "bilinear"
            result = tf.transpose(x, (0, 2, 3, 1))
            result = tf.image.crop_and_resize(
                result,
                boxes=roi,
                box_indices=np.arange(int(x.shape[0])),
                crop_size=sizes,
                method=method,
                extrapolation_value=operation.extrapolation_value,
            )
            result = tf.transpose(result, (0, 3, 1, 2))
            return result

        return resize_func

    def visit_Shape(self, operation: operations.Shape):
        x_ = self.visit(operation.x)

        @self._cached
        def shape_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.shape(x)
            return result

        return shape_func

    def visit_Sigmoid(self, operation: operations.Sigmoid):
        x_ = self.visit(operation.x)

        @self._cached
        def sigmoid_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.nn.sigmoid(x)
            return result

        return sigmoid_func

    def visit_Sign(self, operation: operations.Sign):
        x_ = self.visit(operation.x)

        @self._cached
        def sign_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.math.sign(x)
            return result

        return sign_func

    def visit_Slice(self, operation: operations.Slice):
        x_ = self.visit(operation.x)
        starts_ = self.visit(operation.starts)
        ends_ = self.visit(operation.ends)
        axes_ = self.visit(operation.axes)
        steps_ = self.visit(operation.steps)

        @self._cached
        def slice_func(*inputs):
            x, starts, ends, axes, steps = _concretize(
                [x_, starts_, ends_, axes_, steps_], inputs
            )
            n = x.ndim
            slices = [slice(None) for _ in range(n)]
            if axes is None:
                axes = range(n)
            if steps is None:
                steps = [1 for _ in range(n)]
            for i, axis in enumerate(axes):
                slices[axis] = slice(starts[i], ends[i], steps[i])
            result = x[tuple(slices)]
            return result

        return slice_func

    def visit_Softmax(self, operation: operations.Softmax):
        x_ = self.visit(operation.x)

        @self._cached
        def softmax_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.nn.softmax(x, axis=operation.axis)
            return result

        return softmax_func

    def visit_Split(self, operation: operations.Split):
        x_ = self.visit(operation.x)
        split_ = self.visit(operation.split)

        @self._cached
        def split_func(*inputs):
            x, split = _concretize([x_, split_], inputs)
            x = tf.split(x, tf.convert_to_tensor(split), axis=operation.axis)
            return x

        return split_func

    def visit_Sub(self, operation: operations.Sub):
        a_ = self.visit(operation.a)
        b_ = self.visit(operation.b)

        @self._cached
        def sub_func(*inputs):
            a, b = _concretize([a_, b_], inputs)
            result = tf.subtract(a, b)
            return result

        return sub_func

    def visit_Tanh(self, operation: operations.Tanh):
        x_ = self.visit(operation.x)

        @self._cached
        def tanh_func(*inputs):
            x = _concretize([x_], inputs)
            result = tf.nn.tanh(x)
            return result

        return tanh_func

    def visit_Tile(self, operation: operations.Tile):
        x_ = self.visit(operation.x)
        repeats_ = self.visit(operation.repeats)

        @self._cached
        def tile_func(*inputs):
            x, repeats = _concretize([x_, repeats_], inputs)
            result = tf.tile(x, repeats)
            return result

        return tile_func

    def visit_Transpose(self, operation: operations.Transpose):
        x_ = self.visit(operation.x)

        @self._cached
        def transpose_func(*inputs):
            x = _concretize([x_], inputs)
            permutation = operation.permutation
            result = tf.transpose(x, permutation)
            return result

        return transpose_func

    def visit_Unsqueeze(self, operation: operations.Unsqueeze):
        x_ = self.visit(operation.x)
        axes_ = self.visit(operation.axes)

        @self._cached
        def unsqueeze_func(*inputs):
            x, axes = _concretize([x_, axes_], inputs)
            for axis in sorted(axes):
                x = tf.expand_dims(x, axis)
            return x

        return unsqueeze_func
