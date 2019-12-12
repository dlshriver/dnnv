import numpy as np
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


class ReluplexTranslatorError(VerifierTranslatorError):
    pass


class ReluplexTranslator:
    def __init__(self, dnn, phi):
        dnn = dnn.simplify()
        networks = phi.networks
        if len(networks) == 0:
            raise ReluplexTranslatorError("Property does not use a network")
        if len(networks) > 1:
            raise ReluplexTranslatorError("Property has more than 1 network")
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
            constraint_type, input_bounds, output_constraint = property_extractor.extract(
                conjunction
            )
            self.input_lower_bound = np.asarray(input_bounds[0])
            self.input_upper_bound = np.asarray(input_bounds[1])
            self.layers = as_layers(self.op_graph)
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

                W2 = np.zeros((num_classes - 1, 1))
                for i in range(num_classes - 1):
                    W2[i, 0] = -1
                b2 = np.zeros(1) + 10 * np.finfo(np.float32).resolution
                self.layers.append(FullyConnected(W2, b2))
            elif constraint_type == "regression":
                assert len(output_constraint) == 1
                relation, value = list(output_constraint.items())[0]
                W1 = np.zeros((1, 1))
                b1 = np.zeros((1,))
                if relation in ["<", "<="]:
                    W1[0, 0] = 1
                    b1[0] = -value
                elif relation in [">", ">="]:
                    W1[0, 0] = -1
                    b1[0] = value
                else:
                    raise ReluplexTranslatorError(
                        f"Unsupported property constraint type: (output {relation} {value})"
                    )
                if self.layers[-1].activation is None:
                    self.layers[-1].weights = self.layers[-1].weights @ W1
                    self.layers[-1].bias = self.layers[-1].bias @ W1 + b1
                else:
                    self.layers.append(FullyConnected(W1, b1))
            else:
                raise ReluplexTranslatorError(
                    f"Unsupported property type: {constraint_type}"
                )
            nnet_file = tempfile.NamedTemporaryFile(
                dir=self.tempdir.name, suffix=".nnet", delete=False
            )
            for line in self.as_nnet():
                nnet_file.write(f"{line}\n".encode("utf-8"))
            nnet_file.close()
            yield ReluplexCheck(self, nnet_file.name)

    def ensure_layer_support(self):
        if not isinstance(self.layers[0], InputLayer):
            raise ReluplexTranslatorError(
                f"Unsupported layer type: {type(layer).__name__}"
            )
        layers = self.layers[1:]
        if isinstance(layers[-1], FullyConnected) and layers[-1].activation is not None:
            raise ReluplexTranslatorError("Unsupported output layer")
        for i, layer in enumerate(layers, 1):
            if not isinstance(layer, FullyConnected):
                raise ReluplexTranslatorError(
                    f"Unsupported layer type: {type(layer).__name__}"
                )
            if i < len(layers) and layer.activation != "relu":
                raise ReluplexTranslatorError(
                    f"Unsupported activation: {layer.activation}"
                )

    def as_nnet(self):
        self.ensure_layer_support()
        if self.input_lower_bound is None:
            raise ReluplexTranslatorError(
                "A lower bound must be specified for the input"
            )
        if self.input_upper_bound is None:
            raise ReluplexTranslatorError(
                "An upper bound must be specified for the input"
            )
        layers = self.layers[1:]
        num_layers = len(layers)
        input_size = np.product(self.layers[0].shape)
        output_size = layers[-1].bias.shape[0]
        layer_sizes = [input_size] + list(l.bias.shape[0] for l in layers)
        max_layer_size = max(layer_sizes)
        yield "%s,%s,%s,%s," % (num_layers, input_size, output_size, max_layer_size)
        yield ",".join([str(size) for size in layer_sizes]) + ","
        yield "0,"
        yield ",".join(["%.12f" % m for m in self.input_lower_bound.flatten()]) + ","
        yield ",".join(["%.12f" % m for m in self.input_upper_bound.flatten()]) + ","
        yield ",".join(["0.0" for _ in range(input_size + 1)]) + ","  # input mean
        yield ",".join(["1.0" for _ in range(input_size + 1)]) + ","  # input std
        for layer in layers:
            for i in range(layer.weights.shape[1]):
                yield ",".join("%.12f" % w for w in layer.weights[:, i])
            for b in layer.bias:
                yield "%.12f" % b


class ReluplexCheck:
    def __init__(self, translator, input_path):
        self.input_path = input_path
        self.layers = translator.layers[:]

    def check(self):
        logger = logging.getLogger(__name__)
        args = ["reluplex", self.input_path]
        proc = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE, encoding="utf8")

        output_lines = []
        while proc.poll() is None:
            line = proc.stdout.readline()
            logger.info(line.strip())
            output_lines.append(line)
        output_lines.extend(proc.stdout.readlines())

        solution_found = False
        solution = []
        for line in output_lines:
            if line.startswith("Solution found!"):
                solution_found = True
            if solution_found and line.startswith("input"):
                x = float(line.split()[2].strip("."))
                solution.append(x)
        if solution_found:
            solution = np.asarray(solution).reshape(self.layers[0].shape)

        if solution_found:
            return SAT
        return UNSAT


def verify(dnn, phi):
    translator = ReluplexTranslator(dnn, phi)

    result = UNSAT
    for property_check in translator:
        result |= property_check.check()
    translator.tempdir.cleanup()
    return result
