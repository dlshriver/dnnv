import numpy as np
import tempfile

from typing import Iterable, List, Optional, Type

from dnnv.nn.layers import Convolutional, FullyConnected, InputLayer, Layer
from dnnv.verifiers.common import HyperRectangle, VerifierTranslatorError

from .layers import conv_as_rlv


def as_rlv(
    input_interval: HyperRectangle,
    layers: List[Layer],
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
) -> Iterable[str]:
    if (input_interval.lower_bounds[0] == -np.inf).any():
        raise translator_error("A lower bound must be specified for all inputs")
    if (input_interval.upper_bounds[0] == np.inf).any():
        raise translator_error("An upper bound must be specified for all inputs")
    input_layer = layers[0]
    if not isinstance(input_layer, InputLayer):
        raise translator_error(
            f"Unsupported input layer type: {type(input_layer).__name__!r}"
        )
    curr_layer = []
    for input_index in np.ndindex(input_layer.shape):
        input_index_str = ":".join(str(i) for i in input_index)
        name = f"input:{input_index_str}"
        curr_layer.append(name)
        yield f"Input {name}"
    layer_id = 1
    prev_layer = tuple(curr_layer)
    input_shape = tuple(input_layer.shape)
    seen_fully_connected = False
    for layer in layers[1:]:
        curr_layer = []
        output_shape = []  # type: List[int]
        if isinstance(layer, FullyConnected):
            if layer.activation is None:
                activation = "Linear"
            elif layer.activation == "relu":
                activation = "ReLU"
            else:
                raise translator_error(
                    f"Unsupported activation type: {layer.activation}"
                )
            weights = layer.weights[layer.w_permutation].T
            assert len(weights) == len(layer.bias)
            for i, (W, bias) in enumerate(zip(weights, layer.bias)):
                assert len(W) == len(prev_layer)
                name = f"layer{layer_id}:fc:{i}"
                curr_layer.append(name)
                computation = " ".join(f"{w:.12f} {n}" for w, n in zip(W, prev_layer))
                yield f"{activation} {name} {bias:.12f} {computation}"
            output_shape = [input_shape[0], len(curr_layer)]
        elif isinstance(layer, Convolutional):
            yield from conv_as_rlv(
                layer, str(layer_id), prev_layer, input_shape, curr_layer, output_shape
            )
        elif hasattr(layer, "as_rlv"):
            yield from layer.as_rlv(
                layer_id, prev_layer, input_shape, curr_layer, output_shape
            )
        else:
            raise translator_error(f"Unsupported layer type: {type(layer).__name__}")
        prev_layer = tuple(curr_layer)
        input_shape = tuple(output_shape)
        layer_id += 1
    for input_index in np.ndindex(input_layer.shape):
        input_index_str = ":".join(str(i) for i in input_index)
        name = f"input:{input_index_str}"
        yield f"Assert <= {input_interval.lower_bounds[0][input_index]:.12f} 1.0 {name}"
        yield f"Assert >= {input_interval.upper_bounds[0][input_index]:.12f} 1.0 {name}"
    if len(prev_layer) != 1:
        raise translator_error("More than 1 output node is not currently supported")
    yield f"Assert >= 0.0 1.0 {prev_layer[0]}"


def to_rlv_file(
    input_interval: HyperRectangle,
    layers: List[Layer],
    dirname: Optional[str] = None,
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
) -> str:
    if dirname is None:
        dirname = tempfile.tempdir
    with tempfile.NamedTemporaryFile(
        mode="w+", dir=dirname, suffix=".rlv", delete=False
    ) as rlv_file:
        for line in as_rlv(input_interval, layers, translator_error=translator_error):
            rlv_file.write(f"{line}\n")
        return rlv_file.name
