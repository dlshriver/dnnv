import numpy as np
import tempfile

from typing import Any, Dict, List, Optional

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.executors import VerifierExecutor
from dnnv.verifiers.common.results import SAT, UNSAT, UNKNOWN

from .errors import NnenumError, NnenumTranslatorError


class NnenumExecutor(VerifierExecutor):
    def run(self):
        from nnenum.enumerate import enumerate_network
        from nnenum.onnx_network import load_onnx_network
        from nnenum.settings import Settings
        from nnenum.specification import Specification

        Settings.CHECK_SINGLE_THREAD_BLAS = False
        Settings.PRINT_OUTPUT = False

        onnx_filename, (lb, ub) = self.args
        init_box = np.array(list(zip(lb, ub)), dtype=np.float32)

        network = load_onnx_network(onnx_filename)

        spec = Specification([[1]], [0])

        result = enumerate_network(init_box, network, spec)
        return result


class Nnenum(Verifier):
    translator_error = NnenumTranslatorError
    verifier_error = NnenumError
    executor = NnenumExecutor

    @classmethod
    def is_installed(cls):
        try:
            import nnenum
        except ImportError:
            return False
        return True

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise self.translator_error(
                "Unsupported network: More than 1 input variable"
            )

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".onnx", delete=False
        ) as onnx_model_file:
            prop.suffixed_op_graph().export_onnx(onnx_model_file.name)

        lb = prop.input_constraint.lower_bounds[0].flatten().copy()
        ub = prop.input_constraint.upper_bounds[0].flatten().copy()

        return onnx_model_file.name, (lb, ub)

    def parse_results(self, prop, results):
        result_str = results.result_str
        if result_str == "safe":
            return UNSAT, None
        elif result_str == "unsafe":
            input_shape, input_dtype = prop.op_graph.input_details[0]
            cex = np.array(list(results.cinput)).reshape(input_shape)
            return SAT, None
        raise self.translator_error(f"Unknown verification result: {result_str}")
