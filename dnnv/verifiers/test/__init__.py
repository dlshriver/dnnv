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


def verify(dnn, phi):
    dnn = dnn.simplify()
    phi.networks[0].concretize(dnn)

    property_extractor = ConvexPolytopeExtractor()
    for prop in property_extractor.extract_from(phi):
        print("=====================")
        print("Property fragment")
        print("~~~~~~~~~~~~~~~~~~~~~")
        print("Network:")
        prop.network.value.pprint()
        print("~~~~~~~~~~~~~~~~~~~~~")
        print("Input Constraints:")
        for constraint in prop.input_constraint.constraints:
            print(
                f"{constraint.coefficients} * x[{constraint.indices}] <= {constraint.b}"
            )
        print("~~~~~~~~~~~~~~~~~~~~~")
        print("Output Constraints:")
        for constraint in prop.output_constraint.constraints:
            print(
                f"{constraint.coefficients} * x[{constraint.indices}] <= {constraint.b}"
            )
        print("=====================")

    return UNKNOWN
