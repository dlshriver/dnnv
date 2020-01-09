import numpy as np
import tempfile

from typing import Generator, List, Tuple, Type

from dnnv import logging
from dnnv.nn.layers import Convolutional, FullyConnected, InputLayer, Layer
from dnnv.verifiers.common import (
    Property,
    HyperRectangle,
    ConvexPolytopeExtractor,
    UNKNOWN,
    CommandLineExecutor,
    VerifierTranslatorError,
)


def conv_as_rlv(
    layer: Convolutional,
    layer_id: str,
    prev_layer: Tuple[str, ...],
    input_shape: Tuple[int, int, int, int],
    curr_layer: List[str],
    output_shape: List[int],
) -> Generator[str, None, None]:
    activation = "Linear" if layer.activation is None else "ReLU"
    prev_layer_arr = np.array(prev_layer).reshape(input_shape)

    k_h, k_w = layer.kernel_shape
    p_top, p_left, p_bottom, p_right = layer.pads
    s_h, s_w = layer.strides

    n, in_c, in_h, in_w = input_shape
    out_c = layer.weights.shape[0]
    out_h = int(np.floor(float(in_h - k_h + p_top + p_bottom) / s_h + 1))
    out_w = int(np.floor(float(in_w - k_w + p_left + p_right) / s_w + 1))
    output_shape.extend([n, out_c, out_h, out_w])

    for k in range(out_c):
        for h in range(out_h):
            r = h * s_h - p_top
            for w in range(out_w):
                c = w * s_w - p_left
                name = f"layer{layer_id}:conv:{k}:{h}:{w}"
                curr_layer.append(name)
                partial_computations = []
                for z in range(in_c):
                    for x in range(k_h):
                        for y in range(k_w):
                            if r + x < 0 or r + x >= in_h:
                                continue
                            if c + y < 0 or c + y >= in_w:
                                continue
                            weight = layer.weights[k, z, x, y]
                            in_name = prev_layer_arr[0, z, r + x, c + y]
                            partial_computations.append(f"{weight:.12f} {in_name}")
                computation = " ".join(partial_computations)
                yield f"{activation} {name} {layer.bias[k]:.12f} {computation}"


def as_rlv(
    input_interval: HyperRectangle,
    layers: List[Layer],
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
):
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
    for layer in layers[1:]:
        curr_layer = []
        output_shape = []  # type: List[int]
        if isinstance(layer, FullyConnected):
            activation = "Linear" if layer.activation is None else "ReLU"
            weights = layer.weights[layer.w_permutation].T
            for i, (weights, bias) in enumerate(zip(weights, layer.bias)):
                name = f"layer{layer_id}:fc:{i}"
                curr_layer.append(name)
                computation = " ".join(
                    f"{w:.12f} {n}" for w, n in zip(weights, prev_layer)
                )
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
        yield f"Assert <= {input_interval.lower_bound[input_index]} 1.0 {name}"
        yield f"Assert >= {input_interval.upper_bound[input_index]} 1.0 {name}"
    if len(prev_layer) != 1:
        raise translator_error("More than 1 output node is not currently supported")
    yield f"Assert <= 0.0 1.0 {prev_layer[0]}"


def verify(dnn, phi):
    dnn = dnn.simplify()
    phi.networks[0].concretize(dnn)

    property_extractor = ConvexPolytopeExtractor()
    for prop in property_extractor.extract_from(phi):
        layers = prop.output_constraint.as_layers(prop.network)
        input_interval = prop.input_constraint.as_hyperrectangle()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}/test.rlv", "w+") as rlv_file:
                for line in as_rlv(input_interval, layers):
                    rlv_file.write(f"{line}\n")
            print(f"Running: planet {tmpdir}/test.rlv")
            executor = CommandLineExecutor("planet", f"{tmpdir}/test.rlv")
            out, err = executor.run()
        print(out[-1])

    return UNKNOWN
