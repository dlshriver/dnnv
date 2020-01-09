import numpy as np

from dnnv import logging
from dnnv.nn.layers import Convolutional, FullyConnected, InputLayer
from dnnv.verifiers.common import (
    IOProperty,
    LinIneqPropertyExtractor,
    UNKNOWN,
    as_layers,
    CommandLineExecutor,
    VerifierTranslatorError,
)


def encode_as_layers(prop: IOProperty):
    if any(
        c.type not in [IOProperty.ConstraintType.LT, IOProperty.ConstraintType.LE]
        for c in prop.output_constraints
    ):
        raise VerifierTranslatorError("Only LT and LE constraints are supported")
    layers = as_layers(prop.network.value)
    num_outputs = layers[-1].bias.shape[0]
    num_intermediate_outputs = sum([1 for c in prop.output_constraints])
    W1 = np.zeros((num_outputs, num_intermediate_outputs))
    b1 = np.zeros(num_intermediate_outputs)
    print(prop.output_constraints)
    for n, c in enumerate(prop.output_constraints):
        b1[n] -= c.b
        for i, v in zip(c.idx, c.coefs):
            W1[i, n] = v
    if layers[-1].activation is None:
        layers[-1].weights = layers[-1].weights @ W1
        layers[-1].bias = layers[-1].bias @ W1 + b1
        layers[-1].activation = "relu"
    else:
        layers.append(FullyConnected(W1, b1, activation="relu"))
    W2 = np.zeros((num_intermediate_outputs, 1))
    for i in range(num_intermediate_outputs):
        W2[i, 0] = 1
    b2 = np.zeros((1,))
    layers.append(FullyConnected(W2, b2))
    return layers


def verify(dnn, phi):
    dnn = dnn.simplify()
    phi.networks[0].concretize(dnn)

    property_extractor = LinIneqPropertyExtractor()
    for prop in property_extractor.extract_from(phi):
        layers = encode_as_layers(prop)

    return UNKNOWN
