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
)

from .errors import ReluplexError, ReluplexTranslatorError
from .utils import to_nnet_file


def parse_results(stdout: List[str], stderr: List[str]):
    for line in stdout:
        if line.startswith("Solution found!"):
            return SAT
        elif line.startswith("No Solution"):
            return UNSAT
    raise ReluplexError(f"No verification result found")


def verify(dnn: OperationGraph, phi: Expression):
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
            executor = CommandLineExecutor(
                "reluplex", f"{nnet_file_name}", verifier_error=ReluplexError
            )
            out, err = executor.run()
            result |= parse_results(out, err)
            if result == SAT or result == UNKNOWN:
                return result

    return result
