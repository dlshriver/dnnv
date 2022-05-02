import tempfile
from functools import partial

import numpy as np

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.reductions import HalfspacePolytope, IOPolytopeReduction
from dnnv.verifiers.common.results import SAT, UNKNOWN, UNSAT
from dnnv.verifiers.common.utils import as_layers
from dnnv.verifiers.planet.utils import to_rlv_file

from .errors import BabError, BabTranslatorError


class BaB(Verifier):
    reduction = partial(IOPolytopeReduction, HalfspacePolytope, HalfspacePolytope)
    translator_error = BabTranslatorError
    verifier_error = BabError
    parameters = {
        "reluify_maxpools": Parameter(bool, default=False),
        "smart_branching": Parameter(bool, default=False),
    }

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise self.translator_error(
                "Unsupported network: More than 1 input variable"
            )
        layers = as_layers(
            prop.suffixed_op_graph().simplify(),
            translator_error=self.translator_error,
        )
        rlv_file_path = to_rlv_file(
            prop.input_constraint,
            layers,
            translator_error=self.translator_error,
        )

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".npy", delete=False
        ) as output_file:
            self._tmp_output_file = output_file

        return (
            "bab",
            rlv_file_path,
            "-o",
            self._tmp_output_file.name,
            "--reluify_maxpools",
            str(self.parameter_values["reluify_maxpools"]),
            "--smart_branching",
            str(self.parameter_values["smart_branching"]),
        )

    def parse_results(self, prop, results):
        stdout, _ = results
        result_str = stdout[-1]
        if result_str == "safe":
            return UNSAT, None
        if result_str == "unsafe":
            shape, dtype = prop.op_graph.input_details[0]
            cex = (
                np.load(self._tmp_output_file.name, allow_pickle=True)
                .astype(dtype)
                .reshape(shape)
            )
            return SAT, cex
        if result_str == "unknown":
            return UNKNOWN, None
        raise self.translator_error(f"Unknown verification result: {result_str}")
