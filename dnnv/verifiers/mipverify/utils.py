import numpy as np
import scipy.io
import tempfile

from typing import Dict, IO, Iterable, List, Optional, Type

from dnnv.nn.layers import Convolutional, FullyConnected, InputLayer, Layer
from dnnv.verifiers.common import HyperRectangle, VerifierTranslatorError


def as_mipverify(
    layers: List[Layer],
    weights_file: IO,
    sample_input: np.ndarray,
    linf=None,
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
) -> Iterable[str]:
    yield "using MIPVerify, Gurobi, MAT"
    yield f'parameters = matread("{weights_file.name}")'
    layer_names = []  # type: List[str]
    layer_parameters = {}  # type: Dict[str, np.ndarray]
    input_layer = layers[0]
    if not isinstance(input_layer, InputLayer):
        raise translator_error(
            f"Unsupported input layer type: {type(input_layer).__name__}"
        )
    last_layer = layers[-1]
    if not isinstance(last_layer, FullyConnected):
        raise translator_error(
            f"Unsupported output layer type: {type(last_layer).__name__}"
        )
    elif last_layer.bias.shape[0] == 1:
        last_layer.bias = np.array([-last_layer.bias[0], last_layer.bias[0]])
        last_layer.weights = np.stack(
            [-last_layer.weights[:, 0], last_layer.weights[:, 0]], 1
        )
    shape = np.asarray(input_layer.shape)[[0, 2, 3, 1]]
    seen_fullyconnected = False
    for layer_id, layer in enumerate(layers[1:], 1):
        layer_name = f"layer{layer_id}"
        if isinstance(layer, FullyConnected):
            if layer_id == 1:
                layer_names.append(f"Flatten({sample_input.ndim})")
            elif isinstance(layers[layer_id - 1], Convolutional):
                layer_names.append(f"Flatten(4)")
            weights = layer.weights.astype(np.float32)
            weights = weights[layer.w_permutation]
            if not seen_fullyconnected:
                seen_fullyconnected = True
                if len(shape) == 4:
                    weights = weights[
                        (
                            np.arange(np.product(shape))
                            .reshape(shape[[0, 3, 1, 2]])
                            .transpose((0, 2, 3, 1))
                            .flatten()
                        )
                    ]
            layer_parameters[f"{layer_name}_weight"] = weights
            yield f'{layer_name}_W = parameters["{layer_name}_weight"]'
            if len(layer.bias) > 1:
                layer_parameters[f"{layer_name}_bias"] = layer.bias
                yield f'{layer_name}_b = dropdims(parameters["{layer_name}_bias"], dims=1)'
            else:
                yield f"{layer_name}_b = [{layer.bias.item()}]"
            yield f"{layer_name} = Linear({layer_name}_W, {layer_name}_b)"
            layer_names.append(layer_name)
            if layer.activation == "relu":
                layer_names.append("ReLU()")
            elif layer.activation is not None:
                raise translator_error(
                    f"{layer.activation} activation is currently unsupported"
                )
        elif isinstance(layer, Convolutional):
            if any(s != layer.strides[0] for s in layer.strides):
                raise translator_error(
                    "Different strides in height and width are not supported"
                )
            input_shape = shape.copy()
            shape[3] = layer.weights.shape[0]  # number of output channels
            shape[1] = np.ceil(
                float(input_shape[1]) / float(layer.strides[0])
            )  # output height
            shape[2] = np.ceil(
                float(input_shape[2]) / float(layer.strides[1])
            )  # output width

            pad_along_height = max(
                (shape[1] - 1) * layer.strides[0]
                + layer.kernel_shape[0]
                - input_shape[1],
                0,
            )
            pad_along_width = max(
                (shape[2] - 1) * layer.strides[1]
                + layer.kernel_shape[1]
                - input_shape[2],
                0,
            )
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left

            same_pads = [pad_top, pad_left, pad_bottom, pad_right]
            if any(p1 != p2 for p1, p2 in zip(layer.pads, same_pads)):
                raise translator_error("Non-SAME padding is not supported")
            weights = layer.weights.transpose((2, 3, 1, 0))
            layer_parameters[f"{layer_name}_weight"] = weights
            yield f'{layer_name}_W = parameters["{layer_name}_weight"]'
            if len(layer.bias) > 1:
                layer_parameters[f"{layer_name}_bias"] = layer.bias
                yield f'{layer_name}_b = dropdims(parameters["{layer_name}_bias"], dims=1)'
            else:
                yield f"{layer_name}_b = [{layer.bias.item()}]"
            yield f"{layer_name} = Conv2d({layer_name}_W, {layer_name}_b, {layer.strides[0]})"
            layer_names.append(layer_name)
            if layer.activation == "relu":
                layer_names.append("ReLU()")
            elif layer.activation is not None:
                raise translator_error(
                    f"{layer.activation} activation is currently unsupported"
                )
        elif hasattr(layer, "as_julia"):
            for line in layer.as_julia(layers):
                yield line
        else:
            raise translator_error(f"Unsupported layer type: {type(layer).__name__}")
    yield "nn = Sequential(["
    for layer_name in layer_names:
        yield f"\t{layer_name},"
    yield f'], "{str(weights_file.name)[:-4]}")'
    if sample_input.ndim == 4:
        sample_input = sample_input.transpose((0, 2, 3, 1))
    layer_parameters["input"] = sample_input
    yield f'input = parameters["input"]'
    yield 'print(nn(input), "\\n")'
    perturbation = ""
    if linf is not None:
        perturbation = (
            f", pp=MIPVerify.LInfNormBoundedPerturbationFamily({linf}), norm_order=Inf"
        )
    # class 1 (1-indexed) if property is FALSE
    yield f"d = MIPVerify.find_adversarial_example(nn, input, 1, GurobiSolver(){perturbation}, solve_if_predicted_in_targeted=false, cache_model=false, rebuild=true)"
    yield 'print((d[:PredictedIndex] == 2) ? d[:SolveStatus] : "TRIVIAL", "\\n")'

    scipy.io.savemat(weights_file.name, layer_parameters)


def to_mipverify_inputs(
    input_interval: HyperRectangle,
    layers: List[Layer],
    dirname: Optional[str] = None,
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
) -> Dict[str, str]:
    mipverify_inputs = {}
    if dirname is None:
        dirname = tempfile.tempdir

    lb = np.asarray(input_interval.lower_bound).transpose((0, 2, 3, 1))
    ub = np.asarray(input_interval.upper_bound).transpose((0, 2, 3, 1))
    if np.any(lb < 0) or np.any(ub > 1):
        raise translator_error(
            "Inputs intervals that extend outside of [0, 1] are not supported"
        )
    sample_input = (lb + ub) / 2
    linf = (ub.flatten() - lb.flatten()) / 2
    if not np.allclose(linf, linf[0], atol=1e-5):
        raise translator_error("Multiple epsilon values are not supported")

    with tempfile.NamedTemporaryFile(
        dir=dirname, suffix=".mat", delete=False
    ) as weights_file:
        with tempfile.NamedTemporaryFile(
            mode="w+", dir=dirname, suffix=".jl", delete=False
        ) as prop_file:
            for line in as_mipverify(
                layers,
                weights_file,
                sample_input,
                linf=linf[0],
                translator_error=translator_error,
            ):
                prop_file.write(f"{line}\n")
            mipverify_inputs["property_path"] = prop_file.name
        mipverify_inputs["weights_path"] = weights_file.name

    return mipverify_inputs
