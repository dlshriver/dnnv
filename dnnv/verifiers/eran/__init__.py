import tempfile

import numpy as np

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.results import SAT, UNKNOWN, UNSAT

from .errors import ERANError, ERANTranslatorError


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
        "use_default_heuristic": Parameter(
            bool,
            default=True,
            help="Whether or not to use the ERAN area heuristic.",
        ),
    }

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise self.translator_error(
                "Unsupported network: More than 1 input variable"
            )

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".onnx", delete=False
        ) as onnx_model_file:
            prop.suffixed_op_graph().simplify().export_onnx(
                onnx_model_file.name, add_missing_optional_inputs=True
            )

        input_interval = prop.input_constraint
        spec_lb = input_interval.lower_bounds[0]
        spec_ub = input_interval.upper_bounds[0]
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".npy", delete=False
        ) as input_constraint_file:
            np.save(input_constraint_file.name, (spec_lb, spec_ub))

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".npy", delete=False
        ) as output_file:
            self._tmp_output_file = output_file

        args = (
            "eran",
            onnx_model_file.name,
            input_constraint_file.name,
            "-o",
            self._tmp_output_file.name,
        ) + tuple(
            f"--{k}={v}" for k, v in self.parameter_values.items() if v is not None
        )
        return args

    def parse_results(self, prop, results):
        stdout, stderr = results
        result_str = stdout[-1]
        if result_str == "safe":
            return UNSAT, None
        elif result_str == "unsafe":
            cex = np.load(self._tmp_output_file.name, allow_pickle=True).astype(
                prop.op_graph.input_details[0].dtype
            )
            return SAT, cex
        elif result_str == "unknown":
            return UNKNOWN, None
        raise self.translator_error(f"Unknown verification result: {result_str}")
