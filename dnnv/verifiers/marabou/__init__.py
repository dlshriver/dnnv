import numpy as np
import tempfile

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.results import SAT, UNSAT, UNKNOWN

from .errors import MarabouError, MarabouTranslatorError


class Marabou(Verifier):
    EXE = "marabou.py"
    translator_error = MarabouTranslatorError
    verifier_error = MarabouError

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise self.translator_error(
                "Unsupported network: More than 1 input variable"
            )

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".onnx", delete=False
        ) as onnx_model_file:
            prop.op_graph.export_onnx(onnx_model_file.name)

        lb = prop.input_constraint.lower_bounds[0].copy()
        ub = prop.input_constraint.upper_bounds[0].copy()

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".npy", delete=False
        ) as constraint_file:
            A, b = prop.output_constraint.as_matrix_inequality()
            np.save(constraint_file.name, (lb, ub, (A, b)))

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".npy", delete=False
        ) as output_file:
            self._tmp_output_file = output_file
        args = (
            "marabou.py",
            onnx_model_file.name,
            constraint_file.name,
            "-o",
            self._tmp_output_file.name,
        )
        return args

    def parse_results(self, prop, results):
        result, cinput = np.load(self._tmp_output_file.name, allow_pickle=True)
        if result == False:
            return UNSAT, None
        elif result == True:
            input_shape, input_dtype = prop.op_graph.input_details[0]
            cex = cinput.reshape(input_shape).astype(input_dtype)
            return SAT, cex
        raise self.translator_error(f"Unknown verification result: {result_str}")
