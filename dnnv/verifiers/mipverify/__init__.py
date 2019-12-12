import numpy as np
import scipy.io
import subprocess as sp
import tempfile

from dnnv import logging
from dnnv.nn.layers import Layer, InputLayer, FullyConnected, Convolutional
from dnnv.verifiers.common import SAT, UNSAT, UNKNOWN, PropertyExtractor, as_layers

from .errors import MIPVerifyTranslatorError
from .layers import MIPVERIFY_LAYER_TYPES


class MIPTranslator:
    def __init__(self, dnn, phi):
        dnn = dnn.simplify()
        networks = phi.networks
        if len(networks) == 0:
            raise MIPVerifyTranslatorError("Property does not use a network")
        if len(networks) > 1:
            raise MIPVerifyTranslatorError("Property has more than 1 network")
        network = networks[0]
        network.concretize(dnn)
        self.phi = phi.propagate_constants().to_cnf()
        self.not_phi = ~self.phi
        operation_graph = self.phi.networks[0].concrete_value
        self.op_graph = operation_graph
        self.layers = []
        self.tempdir = tempfile.TemporaryDirectory()

    def __iter__(self):
        property_extractor = PropertyExtractor()
        for conjunction in self.not_phi:
            constraint_type, input_bounds, output_constraint = property_extractor.extract(
                conjunction
            )
            lb = np.asarray(input_bounds[0])
            ub = np.asarray(input_bounds[1])
            # if np.any(lb < 0) or np.any(ub > 1):
            #     raise MIPVerifyTranslatorError(
            #         "MIPVerify does not support inputs outside of the range [0, 1]"
            #     )
            sample_input = (lb + ub) / 2
            linf = (ub.flatten() - lb.flatten()) / 2
            if not np.allclose(linf, linf[0], atol=1e-5):
                raise MIPVerifyTranslatorError(
                    "MIPVerify does not support multiple l-inf values"
                )
            self.layers = as_layers(
                self.op_graph,
                layer_types=[InputLayer, Convolutional, FullyConnected]
                + MIPVERIFY_LAYER_TYPES,
            )
            if constraint_type == "classification-argmax":
                assert len(output_constraint) == 1
                assert "!=" in output_constraint
                c = output_constraint["!="]
                num_classes = self.layers[-1].bias.shape[0]
                W1 = np.zeros((num_classes, num_classes))
                for i in range(num_classes):
                    if i == c:
                        continue
                    W1[i, i] = 1
                    W1[c, i] = -1
                W1 = np.delete(W1, c, axis=1)
                if self.layers[-1].activation is None:
                    self.layers[-1].weights = self.layers[-1].weights @ W1
                    self.layers[-1].bias = self.layers[-1].bias @ W1
                    self.layers[-1].activation = "relu"
                else:
                    b1 = np.zeros(num_classes - 1)
                    self.layers.append(FullyConnected(W1, b1, activation="relu"))

                W2 = np.zeros((num_classes - 1, 2))
                for i in range(num_classes - 1):
                    W2[i, 0] = 1
                b2 = np.zeros(2)
                b2[1] += 10 * np.finfo(np.float32).resolution
                self.layers.append(FullyConnected(W2, b2))
            elif constraint_type == "regression":
                assert len(output_constraint) == 1
                relation, value = list(output_constraint.items())[0]
                W1 = np.zeros((1, 2))
                b1 = np.zeros((2,))
                if relation in ["<", "<="]:
                    W1[0, 1] = 1
                    b1[0] = value
                elif relation in [">", ">="]:
                    W1[0, 0] = 1
                    b1[1] = value
                else:
                    raise MIPVerifyTranslatorError(
                        f"Unsupported property constraint type: (output {relation} {value})"
                    )
                if self.layers[-1].activation is None:
                    self.layers[-1].weights = self.layers[-1].weights @ W1
                    self.layers[-1].bias = self.layers[-1].bias @ W1 + b1
                else:
                    self.layers.append(FullyConnected(W1, b1))
            else:
                raise MIPVerifyTranslatorError(
                    f"Unsupported property type: {constraint_type}"
                )
            julia_file = tempfile.NamedTemporaryFile(
                dir=self.tempdir.name, suffix=".jl", delete=False
            )
            for line in self.as_julia(sample_input, linf=linf[0]):
                julia_file.write(f"{line}\n".encode("utf-8"))
            julia_file.close()
            yield MIPVerifyCheck(self, julia_file.name)

    def as_julia(self, sample_input, linf=None):
        yield "using MIPVerify, Gurobi, MAT"
        weights_file = tempfile.NamedTemporaryFile(
            dir=self.tempdir.name, suffix=".mat", delete=False
        )
        weights_file.close()
        yield f'parameters = matread("{weights_file.name}")'
        layers = []
        layer_parameters = {}
        shape = np.asarray(self.layers[0].shape)[[0, 2, 3, 1]]
        for layer_id, layer in enumerate(self.layers[1:], 1):
            layer_name = f"layer{layer_id}"
            if isinstance(layer, FullyConnected):
                if layer_id == 1:
                    layers.append(f"Flatten({sample_input.ndim})")
                elif isinstance(self.layers[layer_id - 1], Convolutional):
                    layers.append(f"Flatten(4)")
                layer_parameters[f"{layer_name}_weight"] = layer.weights
                yield f'{layer_name}_W = parameters["{layer_name}_weight"]'
                if len(layer.bias) > 1:
                    layer_parameters[f"{layer_name}_bias"] = layer.bias
                    yield f'{layer_name}_b = dropdims(parameters["{layer_name}_bias"], dims=1)'
                else:
                    yield f"{layer_name}_b = [{layer.bias.item()}]"
                yield f"{layer_name} = Linear({layer_name}_W, {layer_name}_b)"
                layers.append(layer_name)
                if layer.activation == "relu":
                    layers.append("ReLU()")
                elif layer.activation is not None:
                    raise MIPVerifyTranslatorError(
                        f"{layer.activation} activation is currently unsupported"
                    )
            elif isinstance(layer, Convolutional):
                if any(s != layer.strides[0] for s in layer.strides):
                    raise MIPVerifyTranslatorError(
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
                    raise MIPVerifyTranslatorError("Non-SAME padding is not supported")
                weights = layer.weights.transpose((2, 3, 1, 0))
                layer_parameters[f"{layer_name}_weight"] = weights
                yield f'{layer_name}_W = parameters["{layer_name}_weight"]'
                if len(layer.bias) > 1:
                    layer_parameters[f"{layer_name}_bias"] = layer.bias
                    yield f'{layer_name}_b = dropdims(parameters["{layer_name}_bias"], dims=1)'
                else:
                    yield f"{layer_name}_b = [{layer.bias.item()}]"
                yield f"{layer_name} = Conv2d({layer_name}_W, {layer_name}_b, {layer.strides[0]})"
                layers.append(layer_name)
                if layer.activation == "relu":
                    layers.append("ReLU()")
                elif layer.activation is not None:
                    raise MIPVerifyTranslatorError(
                        f"{layer.activation} activation is currently unsupported"
                    )
            elif isinstance(layer, tuple(MIPVERIFY_LAYER_TYPES)):
                for line in layer.as_julia(layers):
                    yield line
            else:
                raise MIPVerifyTranslatorError(
                    f"Unsupported layer type for MIPVerify: {type(layer).__name__}"
                )
        yield "nn = Sequential(["
        for layer in layers:
            yield f"\t{layer},"
        yield f'], "{str(weights_file.name)[:-4]}")'
        if sample_input.ndim == 4:
            sample_input = sample_input.transpose((0, 2, 3, 1))
        layer_parameters["input"] = sample_input
        yield f'input = parameters["input"]'
        yield 'print(nn(input), "\\n")'
        perturbation = ""
        if linf is not None:
            perturbation = f", pp=MIPVerify.LInfNormBoundedPerturbationFamily({linf}), norm_order=Inf"
        # class 1 (1-indexed) if property is FALSE
        yield f"d = MIPVerify.find_adversarial_example(nn, input, 1, GurobiSolver(){perturbation}, solve_if_predicted_in_targeted=false, cache_model=false, rebuild=true)"
        yield 'print((d[:PredictedIndex] == 2) ? d[:SolveStatus] : "TRIVIAL", "\\n")'

        scipy.io.savemat(weights_file.name, layer_parameters)


class MIPVerifyCheck:
    def __init__(self, translator, input_path):
        self.input_path = input_path
        self.layers = translator.layers[:]

    def check(self):
        logger = logging.getLogger(__name__)
        args = ["julia", self.input_path]
        proc = sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT, encoding="utf8")

        output_lines = []
        while proc.poll() is None:
            line = proc.stdout.readline()
            logger.info(line.strip())
            output_lines.append(line)
        output_lines.extend(proc.stdout.readlines())
        output_lines = [line for line in output_lines if line]

        result = output_lines[-1].strip().lower()
        if "infeasible" in result:
            return UNSAT
        elif "optimal" in result:
            return SAT
        else:
            raise MIPVerifyTranslatorError(f"Unknown MIPVerify result: {result}")


def verify(dnn, phi):
    translator = MIPTranslator(dnn, phi)

    result = UNSAT
    for mip_property in translator:
        result |= mip_property.check()
    translator.tempdir.cleanup()
    return result
