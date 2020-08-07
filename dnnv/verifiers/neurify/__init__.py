import argparse
import numpy as np
import tempfile

from typing import Any, Dict, List, Optional

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

from .errors import NeurifyError, NeurifyTranslatorError
from .utils import to_neurify_inputs


def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurify.max_depth", default=None, type=int, dest="max_depth")
    parser.add_argument(
        "--neurify.max_thread", default=None, type=int, dest="max_thread"
    )
    return parser.parse_known_args(args)


def parse_results(stdout: List[str], stderr: List[str]):
    if len(stdout) < 2:
        raise NeurifyError(f"Neurify terminated before producing expected output.")
    result = stdout[-2].strip()
    if result == "Falsified.":
        return SAT
    elif result == "Unknown.":
        return UNKNOWN
    elif result == "Proved.":
        return UNSAT
    raise NeurifyError(f"Unexpected verification result: {stdout[-1]}")


def validate_counter_example(prop: Property, stdout: List[str], stderr: List[str]):
    cex_found = False
    input_shape, input_dtype = prop.op_graph.input_details[0]
    for line in stdout:
        if cex_found:
            values = line.split(":")[-1][1:-1].split()
            cex = np.asarray([float(v) for v in values], dtype=input_dtype).reshape(
                input_shape
            )
            break
        if line.endswith("Solution:"):
            cex_found = True
    else:
        lb = prop.input_constraint.lower_bounds[0]
        ub = prop.input_constraint.upper_bounds[0]
        cex = ((lb + ub) / 2).astype(input_dtype)
    if not prop.input_constraint.validate(cex):
        raise NeurifyError("Invalid counter example found: input outside bounds.")
    output = prop.op_graph(cex)
    if not prop.output_constraint.validate(output):
        raise NeurifyError("Invalid counter example found: output outside bounds.")


def verify(dnn: OperationGraph, phi: Expression, **kwargs: Dict[str, Any]):
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
                raise NeurifyTranslatorError(
                    "Unsupported network: More than 1 input variable"
                )
            layers = as_layers(
                prop.suffixed_op_graph(), translator_error=NeurifyTranslatorError,
            )
            neurify_inputs = to_neurify_inputs(
                prop.input_constraint,
                layers,
                dirname=dirname,
                translator_error=NeurifyTranslatorError,
            )
            logger.debug("Running neurify")
            executor = CommandLineExecutor(
                "neurify",
                "-n",
                neurify_inputs["nnet_path"],
                "-x",
                neurify_inputs["input_path"],
                "-sl",
                "0.0000000000001",  # TODO: remove magic number
                "-I",
                neurify_inputs["input_interval_path"],
                "-v",
                *[f"--{k}={v}" for k, v in kwargs.items() if v is not None],
                verifier_error=NeurifyError,
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
