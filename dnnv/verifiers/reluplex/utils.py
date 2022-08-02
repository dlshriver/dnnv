import tempfile
from typing import Iterable, List, Optional, Type

import numpy as np

from dnnv.nn.layers import FullyConnected, InputLayer, Layer
from dnnv.verifiers.common import HyperRectangle, VerifierTranslatorError


def as_reluplex_nnet(
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
    fc_layers = []
    for i, layer in enumerate(layers[1:], 2):
        if not isinstance(layer, FullyConnected):
            raise translator_error(f"Unsupported layer type: {type(layer).__name__}")
        if i < len(layers) and layer.activation != "relu":
            raise translator_error(
                f"Unsupported activation on intermediate layer: {layer.activation}"
            )
        elif i == len(layers) and layer.activation is not None:
            raise translator_error(
                f"Unsupported activation on final layer: {layer.activation}"
            )
        fc_layers.append(layer)

    num_layers = len(fc_layers)
    input_size = np.product(input_layer.shape)
    output_size = fc_layers[-1].bias.shape[0]
    layer_sizes = [input_size] + list(l.bias.shape[0] for l in fc_layers)
    max_layer_size = max(layer_sizes)
    yield "%d,%d,%d,%d," % (
        num_layers,
        input_size,
        output_size,
        max_layer_size,
    )
    yield ",".join([str(size) for size in layer_sizes]) + ","
    yield "0,"
    yield ",".join(
        ["%.12f" % m for m in input_interval.lower_bounds[0].flatten()]
    ) + ","
    yield ",".join(
        ["%.12f" % m for m in input_interval.upper_bounds[0].flatten()]
    ) + ","
    yield ",".join(["0.0" for _ in range(input_size + 1)]) + ","  # input mean
    yield ",".join(["1.0" for _ in range(input_size + 1)]) + ","  # input range
    for layer_num, layer in enumerate(fc_layers, 1):
        weights = layer.weights[layer.w_permutation]
        for i in range(weights.shape[1]):
            yield ",".join("%.12f" % w for w in weights[:, i])
        for b in layer.bias:
            yield "%.12f" % b


def to_nnet_file(
    input_interval: HyperRectangle,
    layers: List[Layer],
    dirname: Optional[str] = None,
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w+", dir=dirname, suffix=".nnet", delete=False
    ) as nnet_file:
        for line in as_reluplex_nnet(
            input_interval, layers, translator_error=translator_error
        ):
            nnet_file.write(f"{line}\n")
        return nnet_file.name
