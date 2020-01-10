import argparse
import tempfile

from typing import Any, Dict, List

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

from .errors import NeurifyError, NeurifyTranslatorError
from .utils import to_neurify_inputs


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurify.max_thread", default=0, type=int, dest="max_thread")
    return parser.parse_known_args(args)


def parse_results(stdout: List[str], stderr: List[str]):
    result = stdout[-2].strip()
    if result == "Falsified.":
        return SAT
    elif result == "Unknown.":
        return UNKNOWN
    elif result == "Proved.":
        return UNSAT
    raise NeurifyError(f"Unexpected verification result: {stdout[-1]}")


def verify(dnn: OperationGraph, phi: Expression, **kwargs: Dict[str, Any]):
    dnn = dnn.simplify()
    phi.networks[0].concretize(dnn)

    result = UNSAT
    property_extractor = ConvexPolytopeExtractor()
    with tempfile.TemporaryDirectory() as dirname:
        dirname = "/home/dshriver/Projects/DNNV/tmp/testing"
        for prop in property_extractor.extract_from(phi):
            layers = prop.output_constraint.as_layers(
                prop.network, translator_error=NeurifyTranslatorError
            )
            input_interval = prop.input_constraint.as_hyperrectangle()
            neurify_inputs = to_neurify_inputs(
                input_interval,
                layers,
                dirname=dirname,
                translator_error=NeurifyTranslatorError,
            )
            epsilon = neurify_inputs["epsilon"]
            executor = CommandLineExecutor(
                "neurify",
                "-n",
                neurify_inputs["nnet_path"],
                "-x",
                neurify_inputs["input_path"],
                "-sl",
                "0.0",
                f"-i={epsilon}",
                "-v",
                verifier_error=NeurifyError,
            )
            out, err = executor.run()
            result |= parse_results(out, err)
            if result == SAT or result == UNKNOWN:
                return result

    return result
