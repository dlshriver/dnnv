import numpy as np
import tempfile

from typing import List, Optional, Type

from dnnv import logging
from dnnv.nn import OperationGraph
from dnnv.properties import Expression
from dnnv.verifiers.common import (
    SAT,
    UNSAT,
    UNKNOWN,
    CommandLineExecutor,
    ConvexPolytopeExtractor,
    Property,
)

from .errors import PlanetError, PlanetTranslatorError
from .layers import PLANET_LAYER_TYPES
from .utils import to_rlv_file


def parse_results(stdout: List[str], stderr: List[str]):
    if len(stdout) == 0:
        raise PlanetError(f"Running planet produced no output.")
    if stdout[-1] == "SAT":
        return SAT
    elif stdout[-1] == "UNSAT":
        return UNSAT
    raise PlanetError(f"Unexpected verification result: {stdout[-1]}")


def validate_counter_example(prop: Property, stdout: List[str], stderr: List[str]):
    shape, dtype = prop.network.value.input_details[0]
    cex = np.zeros(shape, dtype)
    found = False
    for line in stdout:
        if line.startswith("SAT"):
            found = True
        if found and line.startswith("- input"):
            position = tuple(int(i) for i in line.split(":")[1:-1])
            value = float(line.split()[-1])
            cex[position] = value
    for constraint in prop.input_constraint.constraints:
        t = sum(c * cex[i] for c, i in zip(constraint.coefficients, constraint.indices))
        if (t - constraint.b) > 1e-6:
            raise PlanetError("Invalid counter example found: input outside bounds.")
    output = prop.network.value(cex)
    for constraint in prop.output_constraint.constraints:
        t = sum(
            c * output[i] for c, i in zip(constraint.coefficients, constraint.indices)
        )
        if (t - constraint.b) > 1e-6:
            raise PlanetError("Invalid counter example found.")


def verify(dnn: OperationGraph, phi: Expression):
    dnn = dnn.simplify()
    phi.networks[0].concretize(dnn)

    result = UNSAT
    property_extractor = ConvexPolytopeExtractor()
    with tempfile.TemporaryDirectory() as dirname:
        for prop in property_extractor.extract_from(phi):
            layers = prop.output_constraint.as_layers(
                prop.network,
                extra_layer_types=PLANET_LAYER_TYPES,
                translator_error=PlanetTranslatorError,
            )
            input_interval = prop.input_constraint.as_hyperrectangle()
            rlv_file_name = to_rlv_file(
                input_interval,
                layers,
                dirname=dirname,
                translator_error=PlanetTranslatorError,
            )
            executor = CommandLineExecutor(
                "planet", f"{rlv_file_name}", verifier_error=PlanetError
            )
            out, err = executor.run()
            result |= parse_results(out, err)
            if result == SAT:
                validate_counter_example(prop, out, err)
            if result == SAT or result == UNKNOWN:
                return result

    return result
