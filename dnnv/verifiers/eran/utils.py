import numpy as np
import tensorflow.compat.v1 as tf

from typing import List, Type

from dnnv.nn.layers import Convolutional, FullyConnected, InputLayer, Layer
from dnnv.verifiers.common import HyperRectangle, VerifierTranslatorError

from .layers import conv_as_tf


def as_tf(
    layers: List[Layer],
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
    include_input: bool = False,
):
    input_layer = layers[0]
    if not isinstance(input_layer, InputLayer):
        raise translator_error(
            f"Unsupported input layer type: {type(input_layer).__name__!r}"
        )
    input_size = np.asarray(input_layer.shape)
    if len(input_size) == 4:
        input_size = input_size[[0, 2, 3, 1]]
    input_size = [d if d >= 0 else None for d in input_size]
    input_placeholder = x = tf.placeholder(input_layer.dtype, input_size)
    for layer in layers[1:]:
        if isinstance(layer, FullyConnected):
            if len(x.shape) != 2:
                x = tf.reshape(x, (x.shape[0], -1))
            x = tf.nn.bias_add(
                tf.matmul(x, layer.weights.astype(np.float32)),
                layer.bias.astype(np.float32),
            )
            if layer.activation == "relu":
                x = tf.nn.relu(x)
            elif layer.activation is not None:
                raise translator_error(
                    f"{layer.activation} activation is currently unsupported"
                )
        elif isinstance(layer, Convolutional):
            x = conv_as_tf(layer, x)
        elif hasattr(layer, "as_tf"):
            x = layer.as_tf(x)
        else:
            raise translator_error(f"Unsupported layer type: {type(layer).__name__}")
    if include_input:
        return input_placeholder, x
    return x
