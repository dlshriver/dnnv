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
    HalfspacePolytope,
    HalfspacePolytopePropertyExtractor,
    HyperRectangle,
    Property,
    as_layers,
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
    shape, dtype = prop.op_graph.input_details[0]
    cex = np.zeros(np.product(shape), dtype)
    found = False
    for line in stdout:
        if found and line.startswith("input"):
            index = int(line.split("]", maxsplit=1)[0].split("[")[-1])
            cex[index] = float(line.split()[-1][:-1])
        if line.startswith("Solution found!"):
            found = True
    cex = cex.reshape(shape)
    if not prop.input_constraint.validate(cex):
        raise ReluplexError("Invalid counter example found: input outside bounds.")
    output = prop.op_graph(cex)
    if not prop.output_constraint.validate(output):
        raise ReluplexError("Invalid counter example found: output outside bounds.")


def verify(dnn: OperationGraph, phi: Expression):
    logger = logging.getLogger(__name__)
    dnn = dnn.simplify()
    phi.networks[0].concretize(dnn)

    result = UNSAT
    property_extractor = HalfspacePolytopePropertyExtractor(
        HyperRectangle, HalfspacePolytope
    )
    with tempfile.TemporaryDirectory() as dirname:
        for prop in property_extractor.extract_from(~phi):
            if prop.input_constraint.num_variables > 1:
                raise ReluplexTranslatorError(
                    "Unsupported network: More than 1 input variable"
                )
            layers = as_layers(
                prop.suffixed_op_graph(), translator_error=ReluplexTranslatorError,
            )
            nnet_file_name = to_nnet_file(
                prop.input_constraint,
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
