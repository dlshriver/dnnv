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
    HalfspacePolytope,
    HalfspacePolytopePropertyExtractor,
    HyperRectangle,
    Property,
    as_layers,
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
    shape, dtype = prop.op_graph.input_details[0]
    cex = np.zeros(shape, dtype)
    found = False
    for line in stdout:
        if line.startswith("SAT"):
            found = True
        if found and line.startswith("- input"):
            position = tuple(int(i) for i in line.split(":")[1:-1])
            value = float(line.split()[-1])
            cex[position] = value
    cex = cex.reshape(shape)
    # planet output has a precision of 5 decimal places
    if not prop.input_constraint.validate(cex, threshold=1e-5):
        raise PlanetError("Invalid counter example found: input outside bounds.")
    output = prop.op_graph(cex)
    if not prop.output_constraint.validate(output):
        raise PlanetError("Invalid counter example found: output outside bounds.")


def verify(dnn: OperationGraph, phi: Expression):
    dnn = dnn.simplify()
    phi.networks[0].concretize(dnn)

    result = UNSAT
    property_extractor = HalfspacePolytopePropertyExtractor(
        HyperRectangle, HalfspacePolytope
    )
    with tempfile.TemporaryDirectory() as dirname:
        for prop in property_extractor.extract_from(~phi):
            if prop.input_constraint.num_variables > 1:
                raise PlanetTranslatorError(
                    "Unsupported network: More than 1 input variable"
                )
            layers = as_layers(
                prop.suffixed_op_graph(),
                extra_layer_types=PLANET_LAYER_TYPES,
                translator_error=PlanetTranslatorError,
            )
            rlv_file_name = to_rlv_file(
                prop.input_constraint,
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
