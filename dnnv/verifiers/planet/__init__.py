import numpy as np
import subprocess as sp
import tempfile

from dnnv import logging
from dnnv.nn.layers import Convolutional, FullyConnected, InputLayer
from dnnv.verifiers.common import (
    SAT,
    UNSAT,
    UNKNOWN,
    as_layers,
    PropertyExtractor,
    VerifierTranslatorError,
)

from .errors import PlanetTranslatorError
from .layers import PLANET_LAYER_TYPES, conv_as_rlv


class RLVTranslator:
    def __init__(
        self,
        dnn,
        phi,
        property_checker,
        translator_error=VerifierTranslatorError,
        layer_types=None,
    ):
        self.property_checker = property_checker
        self.translator_error = translator_error

        dnn = dnn.simplify()
        networks = phi.networks
        if len(networks) == 0:
            raise self.translator_error("Property does not use a network")
        if len(networks) > 1:
            raise self.translator_error("Property has more than 1 network")
        network = networks[0]
        network.concretize(dnn)
        self.phi = phi.propagate_constants().to_cnf()
        self.not_phi = ~self.phi
        operation_graph = self.phi.networks[0].concrete_value
        self.layers = []
        self.layer_types = layer_types
        self.op_graph = operation_graph
        self.tempdir = tempfile.TemporaryDirectory()

    def __iter__(self):
        property_extractor = PropertyExtractor()
        for conjunction in self.not_phi:
            constraint_type, input_bounds, output_constraint = property_extractor.extract(
                conjunction
            )
            self.input_lower_bound = np.asarray(input_bounds[0])
            self.input_upper_bound = np.asarray(input_bounds[1])
            self.layers = as_layers(self.op_graph, layer_types=self.layer_types)
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
            elif constraint_type == "regression":
                assert len(output_constraint) == 1
                relation, value = list(output_constraint.items())[0]
                W1 = np.zeros((1, 1))
                b1 = np.zeros((1,))
                if relation in ["<", "<="]:
                    W1[0, 0] = -1
                    b1[0] = value
                elif relation in [">", ">="]:
                    W1[0, 0] = 1
                    b1[0] = -value
                else:
                    raise self.translator_error(
                        f"Unsupported property constraint type: (output {relation} {value})"
                    )
                if self.layers[-1].activation is None:
                    self.layers[-1].weights = self.layers[-1].weights @ W1
                    self.layers[-1].bias = self.layers[-1].bias @ W1 + b1
                else:
                    self.layers.append(FullyConnected(W1, b1))
            else:
                raise self.translator_error(
                    f"Unsupported property type: {constraint_type}"
                )
            rlv_file = tempfile.NamedTemporaryFile(
                dir=self.tempdir.name, suffix=".rlv", delete=False
            )
            for line in self.as_rlv():
                rlv_file.write(f"{line}\n".encode("utf-8"))
            rlv_file.close()
            yield self.property_checker(self, rlv_file.name)

    def as_rlv(self):
        if self.input_lower_bound is None:
            raise self.translator_error("A lower bound must be specified for the input")
        if self.input_upper_bound is None:
            raise self.translator_error(
                "An upper bound must be specified for the input"
            )
        input_shape = self.layers[0].shape
        num_inputs = np.product(input_shape)
        curr_layer = []
        for input_index in zip(*np.unravel_index(np.arange(num_inputs), input_shape)):
            input_index_str = ":".join(str(i) for i in input_index)
            name = f"input:{input_index_str}"
            curr_layer.append(name)
            yield f"Input {name}"
        layer_id = 1
        prev_layer = curr_layer
        input_shape = input_shape
        for layer in self.layers[1:]:
            curr_layer = []
            output_shape = []
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
                output_shape = input_shape[0], len(curr_layer)
            elif isinstance(layer, Convolutional):
                for line in conv_as_rlv(
                    layer, layer_id, prev_layer, input_shape, curr_layer, output_shape
                ):
                    yield line
            elif isinstance(layer, tuple(PLANET_LAYER_TYPES)):
                for line in layer.as_rlv(
                    layer_id, prev_layer, input_shape, curr_layer, output_shape
                ):
                    yield line
            else:
                raise self.translator_error(
                    f"Unsupported layer type: {type(layer).__name__}"
                )
            prev_layer = tuple(curr_layer)
            input_shape = tuple(output_shape)
            layer_id += 1
        for input_index in zip(
            *np.unravel_index(np.arange(num_inputs), self.layers[0].shape)
        ):
            input_index_str = ":".join(str(i) for i in input_index)
            name = f"input:{input_index_str}"
            yield f"Assert <= {self.input_lower_bound[input_index]} 1.0 {name}"
            yield f"Assert >= {self.input_upper_bound[input_index]} 1.0 {name}"
        if len(prev_layer) != 1:
            raise self.translator_error(
                "More than 1 output node is not currently supported"
            )
        yield f"Assert <= 0.0 1.0 {prev_layer[0]}"


class PlanetCheck:
    def __init__(self, translator, input_path):
        self.input_path = input_path
        self.layers = translator.layers[:]

    def check(self):
        logger = logging.getLogger(__name__)
        args = ["planet", self.input_path]
        proc = sp.Popen(args, stdout=sp.PIPE, stderr=sp.STDOUT, encoding="utf8")

        output_lines = []
        while proc.poll() is None:
            line = proc.stdout.readline()
            logger.info(line.strip())
            output_lines.append(line)
        output_lines.extend(proc.stdout.readlines())

        solution_found = False
        solution = np.zeros(self.layers[0].shape, dtype=self.layers[0].dtype)
        for line in output_lines:
            if line.startswith("SAT"):
                solution_found = True
            if solution_found and line.startswith("- input"):
                position = tuple(int(i) for i in line.split(":")[1:-1])
                value = float(line.split()[-1])
                solution[position] = value

        if solution_found:
            return SAT
        return UNSAT


PlanetTranslator = lambda dnn, phi: RLVTranslator(
    dnn,
    phi,
    PlanetCheck,
    PlanetTranslatorError,
    layer_types=[InputLayer, Convolutional, FullyConnected] + PLANET_LAYER_TYPES,
)


def verify(dnn, phi):
    translator = PlanetTranslator(dnn, phi)

    result = UNSAT
    for property_check in translator:
        result |= property_check.check()
    translator.tempdir.cleanup()
    return result

