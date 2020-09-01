import numpy as np
import tensorflow.compat.v1 as tf

from typing import Any, Dict, List, Optional

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.executors import VerifierExecutor
from dnnv.verifiers.common.results import SAT, UNSAT, UNKNOWN
from dnnv.verifiers.common.utils import as_layers

from .errors import ERANError, ERANTranslatorError
from .layers import ERAN_LAYER_TYPES
from .utils import as_tf


class ERANExecutor(VerifierExecutor):
    def run(self):
        import eran

        eran_model = eran.ERAN(self.args[0])
        _, nn, nlb, nub = eran_model.analyze_box(*self.args[1:])

        output_lower_bound = np.asarray(nlb[-1])
        output_upper_bound = np.asarray(nub[-1])

        return output_lower_bound, output_upper_bound


class ERAN(Verifier):
    translator_error = ERANTranslatorError
    verifier_error = ERANError
    parameters = {
        "domain": Parameter(
            str,
            default="deepzono",
            choices=["deepzono", "deeppoly", "refinezono", "refinepoly"],
            help="The abstract domain to use.",
        ),
        "timeout_lp": Parameter(
            float, default=1.0, help="Time limit for the LP solver."
        ),
        "timeout_milp": Parameter(
            float, default=1.0, help="Time limit for the MILP solver."
        ),
        "use_area_heuristic": Parameter(
            bool, default=True, help="Whether or not to use the ERAN area heuristic."
        ),
    }
    executor = ERANExecutor

    @classmethod
    def is_installed(cls):
        try:
            import eran
        except ImportError:
            return False
        return True

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise self.translator_error(
                "Unsupported network: More than 1 input variable"
            )
        layers = as_layers(
            prop.suffixed_op_graph(),
            extra_layer_types=ERAN_LAYER_TYPES,
            translator_error=self.translator_error,
        )
        input_interval = prop.input_constraint

        spec_lb = input_interval.lower_bounds[0]
        spec_ub = input_interval.upper_bounds[0]
        if len(spec_lb.shape) == 4:
            spec_lb = spec_lb.transpose((0, 2, 3, 1))
            spec_ub = spec_ub.transpose((0, 2, 3, 1))
        g = tf.Graph()
        with g.as_default():
            tf_graph = as_tf(layers, translator_error=self.translator_error)
        return (
            tf_graph,
            spec_lb.flatten().copy(),
            spec_ub.flatten().copy(),
            self.parameters["domain"],
            self.parameters["timeout_lp"],
            self.parameters["timeout_milp"],
            self.parameters["use_area_heuristic"],
        )

    def parse_results(self, prop, results):
        lb, ub = results
        if lb > 0:
            return UNSAT, None
        elif ub <= 0:
            # all inputs violate property, choose center
            input_shape, input_dtype = prop.op_graph.input_details[0]
            lb = prop.input_constraint.lower_bounds[0]
            ub = prop.input_constraint.upper_bounds[0]
            cex = ((lb + ub) / 2).astype(input_dtype)
            return SAT, cex
        return UNKNOWN, None
