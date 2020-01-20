import argparse
import numpy as np
import tensorflow.compat.v1 as tf

from eran import ERAN
from typing import Any, Dict, List, Optional

from dnnv import logging
from dnnv.nn import OperationGraph
from dnnv.properties import Expression
from dnnv.verifiers.common import (
    SAT,
    UNSAT,
    UNKNOWN,
    ConvexPolytope,
    ConvexPolytopeExtractor,
)

from .errors import ERANError, ERANTranslatorError
from .layers import ERAN_LAYER_TYPES, conv_as_tf
from .utils import as_tf


def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eran.domain",
        default="deeppoly",
        choices=["deepzono", "deeppoly", "refinezono", "refinepoly"],
        dest="domain",
    )
    parser.add_argument("--eran.timeout_lp", default=1.0, type=float, dest="timeout_lp")
    parser.add_argument(
        "--eran.timeout_milp", default=1.0, type=float, dest="timeout_milp"
    )
    parser.add_argument(
        "--eran.dont_use_area_heuristic",
        action="store_false",
        dest="use_area_heuristic",
    )
    return parser.parse_known_args(args)


def check(lb, ub):
    if lb > 0:
        return UNSAT
    elif ub <= 0:
        return SAT
    return UNKNOWN


def verify(
    dnn: OperationGraph,
    phi: Expression,
    domain="deeppoly",
    timeout_lp=1.0,
    timeout_milp=1.0,
    use_area_heuristic=True,
    **kwargs: Dict[str, Any]
):
    logger = logging.getLogger(__name__)
    dnn = dnn.simplify()
    phi.networks[0].concretize(dnn)

    result = UNSAT
    property_extractor = ConvexPolytopeExtractor()
    for prop in property_extractor.extract_from(phi):
        with tf.Session(graph=tf.Graph()) as tf_session:
            layers = prop.output_constraint.as_layers(
                prop.network,
                extra_layer_types=ERAN_LAYER_TYPES,
                translator_error=ERANTranslatorError,
            )
            input_interval = prop.input_constraint.as_hyperrectangle()

            spec_lb = input_interval.lower_bound
            spec_ub = input_interval.upper_bound
            if len(spec_lb.shape) == 4:
                spec_lb = spec_lb.transpose((0, 2, 3, 1))
                spec_ub = spec_ub.transpose((0, 2, 3, 1))
            eran_model = ERAN(
                as_tf(layers, translator_error=ERANTranslatorError), session=tf_session
            )
            _, nn, nlb, nub = eran_model.analyze_box(
                spec_lb.flatten().copy(),
                spec_ub.flatten().copy(),
                domain,
                timeout_lp,
                timeout_milp,
                use_area_heuristic,
                **kwargs
            )
            output_lower_bound = np.asarray(nlb[-1])
            output_upper_bound = np.asarray(nub[-1])
            logger.debug("output lower bound: %s", output_lower_bound)
            logger.debug("output upper bound: %s", output_upper_bound)
            result |= check(output_lower_bound, output_upper_bound)
        if result == SAT or result == UNKNOWN:
            return result

    return result
