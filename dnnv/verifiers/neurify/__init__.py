import numpy as np
import os
import subprocess as sp
import tempfile

from ... import logging
from ..common import (
    SAT,
    UNSAT,
    UNKNOWN,
    PropertyExtractor,
    VerifierError,
    VerifierTranslatorError,
    as_layers,
)
from ...nn.layers import Convolutional, FullyConnected, InputLayer


class NeurifyTranslatorError(VerifierTranslatorError):
    pass


class NeurifyTranslator:
    def __init__(self, dnn, phi):
        dnn = dnn.simplify()
        networks = phi.networks
        if len(networks) == 0:
            raise NeurifyTranslatorError("Property does not use a network")
        if len(networks) > 1:
            raise NeurifyTranslatorError("Property has more than 1 network")
        network = networks[0]
        network.concretize(dnn)
        self.phi = phi.propagate_constants().to_cnf()
        self.not_phi = ~self.phi
        operation_graph = self.phi.networks[0].concrete_value
        self.layers = []
        self.op_graph = operation_graph
        self.property_checks = {}
        self.tempdir = tempfile.TemporaryDirectory()

    def __iter__(self):
        property_extractor = PropertyExtractor()
        for conjunction in self.not_phi:
            constraint_type, (
                lower_bound,
                upper_bound,
            ), output_constraint = property_extractor.extract(conjunction)
            input_bounds = (tuple(lower_bound.flatten()), tuple(upper_bound.flatten()))
            property_check = (constraint_type, input_bounds)
            if property_check not in self.property_checks:
                self.property_checks[property_check] = {
                    "input_bounds": (lower_bound, upper_bound),
                    "output_constraints": [],
                }
            self.property_checks[property_check]["output_constraints"].append(
                output_constraint
            )
        for ((constraint_type, _), constraint) in self.property_checks.items():
            self.layers = as_layers(self.op_graph)

            input_bounds = constraint["input_bounds"]
            lb = np.asarray(input_bounds[0])
            ub = np.asarray(input_bounds[1])
            sample_input = (lb + ub) / 2
            linf = (ub.flatten() - lb.flatten()) / 2
            if not np.allclose(linf, linf[0], atol=1e-5):
                raise NeurifyTranslatorError(
                    "Neurify does not support multiple l-inf values"
                )
            epsilon = linf[0]

            output_constraints = {
                key: value
                for c in constraint["output_constraints"]
                for key, value in c.items()
            }
            output_lb = np.zeros(self.layers[-1].bias.shape[0]) - np.inf
            output_ub = np.zeros(self.layers[-1].bias.shape[0]) + np.inf
            if constraint_type == "regression":
                output_ub = np.maximum(
                    output_constraints.get(">", output_lb),
                    output_constraints.get(">=", output_lb),
                )
                output_lb = np.minimum(
                    output_constraints.get("<", output_ub),
                    output_constraints.get("<=", output_ub),
                )
            elif constraint_type == "classification-argmax":
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
                b1 = np.zeros(num_classes - 1)
                if self.layers[-1].activation is None:
                    self.layers[-1].weights = self.layers[-1].weights @ W1
                    self.layers[-1].bias = self.layers[-1].bias @ W1 + b1
                    self.layers[-1].activation = "relu"
                else:
                    self.layers.append(FullyConnected(W1, b1, activation="relu"))

                W2 = np.zeros((num_classes - 1, 1))
                for i in range(num_classes - 1):
                    W2[i, 0] = 1
                b2 = np.zeros((1,))
                self.layers.append(FullyConnected(W2, b2))
                output_ub = np.array([0.0])
                output_lb = np.array([-1000000000.0])
            else:
                raise NeurifyTranslatorError(
                    f"Unsupported property type: {constraint_type}"
                )

            nnet_file = tempfile.NamedTemporaryFile(
                dir=self.tempdir.name, suffix=".nnet", delete=False
            )
            for line in self.as_nnet():
                nnet_file.write(f"{line}\n".encode("utf-8"))
            nnet_file.close()

            input_file = tempfile.NamedTemporaryFile(
                dir=self.tempdir.name, suffix=".input", delete=False
            )
            if len(sample_input.shape) == 4:
                sample_input = sample_input.transpose(0, 1, 3, 2)
            if sample_input.shape[0] != 1:
                raise NeurifyTranslatorError(
                    "Batch sizes greater than 1 are not supported"
                )
            input_file.write(
                ",".join(f"{x:.3f}" for x in sample_input.flatten()).encode("utf-8")
            )
            input_file.close()

            output_constraint_file = tempfile.NamedTemporaryFile(
                dir=self.tempdir.name, suffix=".output", delete=False
            )
            output_constraint_file.write(
                ",".join(f"{x:.12f}" for x in output_lb.flatten()).encode("utf-8")
            )
            output_constraint_file.write(b"\n")
            output_constraint_file.write(
                ",".join(f"{x:.12f}" for x in output_ub.flatten()).encode("utf-8")
            )
            output_constraint_file.write(b"\n")
            output_constraint_file.close()

            yield NeurifyCheck(
                nnet_file.name, input_file.name, output_constraint_file.name, epsilon
            )

    def ensure_layer_support(self):
        if not isinstance(self.layers[0], InputLayer):
            raise NeurifyTranslatorError(
                f"Unsupported layer type: {type(layer).__name__}"
            )
        layers = self.layers[1:]
        if isinstance(layers[-1], FullyConnected) and layers[-1].activation is not None:
            raise NeurifyTranslatorError("Unsupported output layer")
        seen_fully_connected = False
        for i, layer in enumerate(layers, 1):
            if not isinstance(layer, (Convolutional, FullyConnected)):
                raise NeurifyTranslatorError(
                    f"Unsupported layer type: {type(layer).__name__}"
                )
            if isinstance(layer, FullyConnected):
                seen_fully_connected = True
            elif isinstance(layer, Convolutional) and seen_fully_connected:
                raise NeurifyTranslatorError(
                    "Unsupported layer order: FullyConnected before Convolutional"
                )
            if i < len(layers) and layer.activation != "relu":
                raise NeurifyTranslatorError(
                    f"Unsupported activation: {layer.activation}"
                )
            if isinstance(layer, Convolutional):
                if any(k != layer.kernel_shape[0] for k in layer.kernel_shape[1:]):
                    raise NeurifyTranslatorError(
                        "Neurify only supports square convolution kernels"
                    )
                if any(s != layer.strides[0] for s in layer.strides[1:]):
                    raise NeurifyTranslatorError(
                        "Neurify only supports equal strides for height and width"
                    )
                if layer.pads[0] != layer.pads[1]:
                    raise NeurifyTranslatorError(
                        "Neurify only supports padding top and left equally"
                    )
                if layer.pads[2] != layer.pads[3]:
                    raise NeurifyTranslatorError(
                        "Neurify only supports padding bottom and right equally"
                    )
                if (
                    layer.pads[2] - layer.pads[0] < 0
                    or layer.pads[3] - layer.pads[1] < 0
                ):
                    raise NeurifyTranslatorError(
                        "Neurify only supports excess padding for bottom or right edges"
                    )
                if (
                    layer.pads[2] - layer.pads[0] > 1
                    or layer.pads[3] - layer.pads[1] > 1
                ):
                    raise NeurifyTranslatorError(
                        "Neurify only supports excess padding for bottom or right edges of at most 1"
                    )

    def as_nnet(self):
        self.ensure_layer_support()
        layers = self.layers[1:]
        num_layers = len(layers)
        input_size = np.product(self.layers[0].shape)
        output_size = layers[-1].bias.shape[0]

        layer_sizes = [input_size]
        conv_info = []
        shape = np.asarray(self.layers[0].shape)[[0, 2, 3, 1]]
        if shape[0] != 1:
            raise NeurifyTranslatorError("Batch sizes greater than 1 are not supported")
        for layer in layers:
            if isinstance(layer, Convolutional):
                shape[3] = layer.weights.shape[0]  # number of output channels
                shape[1] = np.ceil(
                    float(shape[1] - layer.kernel_shape[0] + 2 * layer.pads[0] + 1)
                    / float(layer.strides[0])
                )  # output height
                shape[2] = np.ceil(
                    float(shape[2] - layer.kernel_shape[1] + 2 * layer.pads[1] + 1)
                    / float(layer.strides[1])
                )  # output width
                if shape[1] != shape[2]:
                    raise NeurifyTranslatorError(
                        "Neurify only supports square inputs for convolution"
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
            else:
                layer_sizes.append(layer.bias.shape[0])
        max_layer_size = max(layer_sizes)

        yield "%s,%s,%s,%s," % (num_layers, input_size, output_size, max_layer_size)
        yield ",".join(str(size) for size in layer_sizes) + ","
        layer_type = lambda l: "1" if isinstance(l, Convolutional) else "0"
        yield ",".join(layer_type(layer) for layer in layers) + ","
        for info in conv_info:
            yield ",".join(str(i) for i in info)

        seen_fully_connected = False
        for layer in layers:
            if isinstance(layer, FullyConnected):
                weights = layer.weights
                weights = layer.weights[layer.w_permutation]
                if not seen_fully_connected:
                    seen_fully_connected = True
                    weights_permutation = (
                        np.arange(np.product(shape[1:]))
                        .reshape(shape)
                        .transpose((0, 3, 2, 1))
                        .flatten()
                    )
                    weights = layer.weights[weights_permutation]
                for i in range(weights.shape[1]):
                    yield ",".join("%.12f" % w for w in weights[:, i])
                for b in layer.bias:
                    yield "%.12f" % b
            elif isinstance(layer, Convolutional):
                for kernel_weights in layer.weights:
                    weights = kernel_weights.transpose(0, 2, 1).flatten()
                    yield ",".join("%.12f" % w for w in weights)
                for b in layer.bias:
                    yield "%.12f" % b


class NeurifyCheck:
    def __init__(self, nnet_path, input_path, output_constraint_path, epsilon):
        self.nnet_path = nnet_path
        self.input_path = input_path
        self.output_constraint_path = output_constraint_path
        self.epsilon = epsilon

    def check(self):
        logger = logging.getLogger(__name__)
        args = [
            "stdbuf",
            "-oL",
            "-eL",
            "neurify",
            "-n",
            self.nnet_path,
            "-x",
            self.input_path,
            "-o",
            self.output_constraint_path,
            f"--linf={self.epsilon}",
            f"-v",
        ]
        proc = sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT, encoding="utf8")

        output_lines = []
        while proc.poll() is None:
            line = proc.stdout.readline()
            logger.info(line.strip())
            output_lines.append(line)
        output_lines.extend(proc.stdout.readlines())
        output_lines = [line for line in output_lines if line]

        result = output_lines[-2].strip()
        if result == "Falsified.":
            return SAT
        elif result == "Unknown.":
            return UNKNOWN
        elif result == "Proved.":
            return UNSAT
        raise NeurifyTranslatorError(f"Unknown property check result: {result}")


def verify(dnn, phi):
    translator = NeurifyTranslator(dnn, phi)

    result = UNSAT
    for property_check in translator:
        result |= property_check.check()
    translator.tempdir.cleanup()
    return result
