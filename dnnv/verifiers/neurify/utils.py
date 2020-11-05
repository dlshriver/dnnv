import numpy as np
import tempfile

from typing import Dict, Iterable, List, Optional, Type, Union

from dnnv.nn.layers import Convolutional, FullyConnected, InputLayer, Layer
from dnnv.verifiers.common.errors import VerifierTranslatorError
from dnnv.verifiers.common.reductions import HalfspacePolytope


def as_neurify_nnet(
    layers: List[Layer],
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
) -> Iterable[str]:
    layer_sizes = []
    conv_info = []
    shape = None  # type: Optional[List[int]]
    seen_fully_connected = False
    for i, layer in enumerate(layers):
        if (
            0 < i < len(layers) - 1
            and isinstance(layer, (Convolutional, FullyConnected))
            and layer.activation != "relu"
        ):
            raise translator_error(f"Unsupported activation: {layer.activation}")
        elif (
            i == len(layers) - 1
            and isinstance(layer, (Convolutional, FullyConnected))
            and layer.activation is not None
        ):
            raise translator_error(f"Unsupported final activation: {layer.activation}")
        if isinstance(layer, InputLayer):
            if shape is not None:
                raise translator_error("Only one InputLayer is supported per network")
            if len(layer.shape) == 4:
                shape = list(np.asarray(layer.shape)[[0, 2, 3, 1]])  # uses NHWC format
            elif len(layer.shape) == 2:
                shape = layer.shape
            elif len(layer.shape) == 1:
                shape = layer.shape
            else:
                raise translator_error(
                    "Unsupported number of axes for network input."
                    f" Expected 4 or 2, got {len(layer.shape)}"
                )
            if len(shape) > 1 and shape[0] != 1:
                raise translator_error("Batch sizes greater than 1 are not supported")
            layer_sizes.append(np.product(layer.shape))
        elif isinstance(layer, Convolutional):
            if seen_fully_connected:
                raise translator_error(
                    "Unsupported layer order: FullyConnected before Convolutional"
                )
            if not isinstance(shape, list):
                raise translator_error("Networks must begin with an InputLayer")
            shape[3] = int(layer.weights.shape[0])  # number of output channels
            shape[1] = int(
                np.ceil(
                    float(
                        shape[1]
                        - layer.kernel_shape[0]
                        + layer.pads[0]
                        + layer.pads[2]
                        + 1
                    )
                    / float(layer.strides[0])
                )
            )  # output height
            shape[2] = int(
                np.ceil(
                    float(
                        shape[2]
                        - layer.kernel_shape[1]
                        + layer.pads[1]
                        + layer.pads[3]
                        + 1
                    )
                    / float(layer.strides[1])
                )
            )  # output width
            if any(s != layer.strides[0] for s in layer.strides[1:]):
                raise translator_error(
                    "Neurify NNET format only supports square inputs for convolution"
                )
            if any(k != layer.kernel_shape[0] for k in layer.kernel_shape[1:]):
                raise translator_error(
                    "Neurify NNET format only supports square kernels for convolution"
                )
            if layer.strides[0] != layer.strides[1]:
                raise translator_error(
                    "Neurify NNET format only supports equal strides for height and width"
                )
            if layer.pads[0] != layer.pads[1]:
                raise translator_error(
                    "Neurify NNET format only supports padding top and left equally"
                )
            if layer.pads[2] != layer.pads[3]:
                raise translator_error(
                    "Neurify NNET format only supports padding bottom and right equally"
                )
            if layer.pads[2] - layer.pads[0] < 0 or layer.pads[3] - layer.pads[1] < 0:
                raise translator_error(
                    "Neurify NNET format only supports excess padding for bottom or right edges"
                )
            if layer.pads[2] - layer.pads[0] > 1 or layer.pads[3] - layer.pads[1] > 1:
                raise translator_error(
                    "Neurify NNET format only supports excess padding for bottom or right edges of at most 1"
                )
            layer_sizes.append(np.product(shape))
            conv_info.append(
                (
                    shape[3],
                    layer.weights.shape[1],
                    layer.kernel_shape[0],
                    layer.strides[0],
                    layer.pads[0],
                    layer.pads[2],
                )
            )
        elif isinstance(layer, FullyConnected):
            seen_fully_connected = True
            layer_sizes.append(layer.bias.shape[0])
        else:
            raise translator_error(f"Unsupported layer type: {type(layer).__name__!r}")
    if shape is None:
        raise translator_error("Networks must begin with an InputLayer")

    yield "%s,%s,%s,%s," % (
        len(layers) - 1,
        layer_sizes[0],
        layer_sizes[-1],
        max(layer_sizes),
    )
    yield ",".join(str(size) for size in layer_sizes) + ","
    layer_type = lambda l: "1" if isinstance(l, Convolutional) else "0"
    yield ",".join(layer_type(layer) for layer in layers[1:]) + ","
    for info in conv_info:
        yield ",".join(str(i) for i in info)

    seen_fully_connected = False
    for layer in layers[1:]:
        if isinstance(layer, FullyConnected):
            weights = layer.weights[layer.w_permutation]
            for i in range(weights.shape[1]):
                yield ",".join("%.12f" % w for w in weights[:, i])
            for b in layer.bias:
                yield "%.12f" % b
        elif isinstance(layer, Convolutional):
            for kernel_weights in layer.weights:
                weights = kernel_weights.flatten()
                yield ",".join("%.12f" % w for w in weights)
            for b in layer.bias:
                yield "%.12f" % b


def to_neurify_inputs(
    input_constraint: HalfspacePolytope,
    layers: List[Layer],
    dirname: Optional[str] = None,
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
) -> Dict[str, str]:
    neurify_inputs = {}

    lb, ub = input_constraint.as_bounds()
    sample_input = (lb + ub) / 2
    with tempfile.NamedTemporaryFile(
        mode="w+", dir=dirname, suffix=".interval", delete=False
    ) as input_file:
        input_file.write(",".join(f"{x:.12f}" for x in lb.flatten()) + "\n")
        input_file.write(",".join(f"{x:.12f}" for x in ub.flatten()) + "\n")
        neurify_inputs["input_interval_path"] = input_file.name

    with tempfile.NamedTemporaryFile(
        mode="w+", dir=dirname, suffix=".hpoly", delete=False
    ) as input_file:
        A, b = input_constraint.as_matrix_inequality()
        input_file.write(f"{A.shape[0]},\n")
        for row in A:
            row_str = ",".join(f"{value:.12f}" for value in row)
            input_file.write(f"{row_str},\n")
        for value in b:
            input_file.write(f"{value:.12f},\n")
        neurify_inputs["input_hpoly_path"] = input_file.name

    with tempfile.NamedTemporaryFile(
        mode="w+", dir=dirname, suffix=".input", delete=False
    ) as input_file:
        if len(sample_input.shape) > 1 and sample_input.shape[0] != 1:
            raise translator_error("Batch sizes greater than 1 are not supported")
        input_file.write(",".join(f"{x:.12f}" for x in sample_input.flatten()))
        neurify_inputs["input_path"] = input_file.name

    with tempfile.NamedTemporaryFile(
        mode="w+", dir=dirname, suffix=".nnet", delete=False
    ) as nnet_file:
        for line in as_neurify_nnet(layers, translator_error=translator_error):
            nnet_file.write(f"{line}\n")
        neurify_inputs["nnet_path"] = nnet_file.name

    return neurify_inputs
