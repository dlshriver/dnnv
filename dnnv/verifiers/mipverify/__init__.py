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
    ConvexPolytopeExtractor,
    Property,
)

from .errors import MIPVerifyError, MIPVerifyTranslatorError
from .layers import MIPVERIFY_LAYER_TYPES
from .utils import to_mipverify_inputs


def parse_results(stdout: List[str], stderr: List[str]):
    result = stdout[-1].lower()
    if "infeasible" in result:
        return UNSAT
    elif "optimal" in result:
        return SAT
    raise MIPVerifyTranslatorError(f"Unexpected verification result: {stdout[-1]}")


def verify(dnn: OperationGraph, phi: Expression, **kwargs: Dict[str, Any]):
    logger = logging.getLogger(__name__)
    dnn = dnn.simplify()
    phi.networks[0].concretize(dnn)

    result = UNSAT
    property_extractor = ConvexPolytopeExtractor()
    with tempfile.TemporaryDirectory() as dirname:
        for prop in property_extractor.extract_from(phi):
            layers = prop.output_constraint.as_layers(
                prop.network,
                extra_layer_types=MIPVERIFY_LAYER_TYPES,
                translator_error=MIPVerifyTranslatorError,
            )
            input_interval = prop.input_constraint.as_hyperrectangle()
            mipverify_inputs = to_mipverify_inputs(
                input_interval,
                layers,
                dirname=dirname,
                translator_error=MIPVerifyTranslatorError,
            )
            logger.debug("Running mipverify")
            executor = CommandLineExecutor(
                "julia",
                mipverify_inputs["property_path"],
                verifier_error=MIPVerifyError,
            )
            out, err = executor.run()
            logger.debug("Parsing results")
            result |= parse_results(out, err)
            if result == SAT or result == UNKNOWN:
                return result

    return result
