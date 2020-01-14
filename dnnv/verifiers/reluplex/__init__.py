import numpy as np
import tempfile

from typing import List

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

from .errors import ReluplexError, ReluplexTranslatorError
from .utils import to_nnet_file


def parse_results(stdout: List[str], stderr: List[str]):
    for line in stdout:
        if line.startswith("Solution found!"):
            return SAT
        elif line.startswith("Can't solve!"):
            return UNSAT
    raise ReluplexError(f"No verification result found")


def validate_counter_example(prop: Property, stdout: List[str], stderr: List[str]):
    shape, dtype = prop.network.value.input_details[0]
    cex = np.zeros(np.product(shape), dtype)
    found = False
    for line in stdout:
        if found and line.startswith("input"):
            index = int(line.split("]", maxsplit=1)[0].split("[")[-1])
            cex[index] = float(line.split()[-1][:-1])
        if line.startswith("Solution found!"):
            found = True
    cex = cex.reshape(shape)
    for constraint in prop.input_constraint.constraints:
        t = sum(c * cex[i] for c, i in zip(constraint.coefficients, constraint.indices))
        if (t - constraint.b) > 1e-6:
            raise ReluplexError("Invalid counter example found: input outside bounds.")
    output = prop.network.value(cex)
    for constraint in prop.output_constraint.constraints:
        t = sum(
            c * output[i] for c, i in zip(constraint.coefficients, constraint.indices)
        )
        if (t - constraint.b) > 1e-6:
            raise ReluplexError("Invalid counter example found.")


def verify(dnn: OperationGraph, phi: Expression):
    logger = logging.getLogger(__name__)
    dnn = dnn.simplify()
    phi.networks[0].concretize(dnn)

    result = UNSAT
    property_extractor = ConvexPolytopeExtractor()
    with tempfile.TemporaryDirectory() as dirname:
        for prop in property_extractor.extract_from(phi):
            layers = prop.output_constraint.as_layers(
                prop.network, translator_error=ReluplexTranslatorError
            )
            input_interval = prop.input_constraint.as_hyperrectangle()
            nnet_file_name = to_nnet_file(
                input_interval,
                layers,
                dirname=dirname,
                translator_error=ReluplexTranslatorError,
            )
            logger.debug("Running reluplex")
            executor = CommandLineExecutor(
                "reluplex", f"{nnet_file_name}", verifier_error=ReluplexError
            )
            out, err = executor.run()
            logger.debug("Parsing results")
            result |= parse_results(out, err)
            if result == SAT:
                logger.debug("SAT! Validating counter example.")
                validate_counter_example(prop, out, err)
            if result == SAT or result == UNKNOWN:
                return result

    return result
