import numpy as np
import os
import tempfile

from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Optional

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.reductions import IOPolytopeReduction, HalfspacePolytope
from dnnv.verifiers.common.results import SAT, UNSAT, UNKNOWN

from .errors import NnenumError, NnenumTranslatorError


class Nnenum(Verifier):
    EXE = "nnenum.py"
    reduction = partial(IOPolytopeReduction, HalfspacePolytope, HalfspacePolytope)
    translator_error = NnenumTranslatorError
    verifier_error = NnenumError
    parameters = {
        "num_processes": Parameter(int, help="Maximum number of processes to use."),
    }

    @contextmanager
    def contextmanager(self):
        orig_OPENBLAS_NUM_THREADS = os.getenv("OPENBLAS_NUM_THREADS")
        orig_OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS")
        try:
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"
            yield
        finally:
            if orig_OPENBLAS_NUM_THREADS is not None:
                os.environ["OPENBLAS_NUM_THREADS"] = orig_OPENBLAS_NUM_THREADS
            else:
                del os.environ["OPENBLAS_NUM_THREADS"]
            if orig_OMP_NUM_THREADS is not None:
                os.environ["OMP_NUM_THREADS"] = orig_OMP_NUM_THREADS
            else:
                del os.environ["OMP_NUM_THREADS"]

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise self.translator_error(
                "Unsupported network: More than 1 input variable"
            )

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".onnx", delete=False
        ) as onnx_model_file:
            prop.op_graph.export_onnx(onnx_model_file.name)

        lb, ub = prop.input_constraint.as_bounds()
        A_in, b_in = prop.input_constraint.as_matrix_inequality()
        A_out, b_out = prop.output_constraint.as_matrix_inequality()

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".npy", delete=False
        ) as constraint_file:
            np.save(constraint_file.name, ((lb, ub), (A_in, b_in), (A_out, b_out)))

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".npy", delete=False
        ) as output_file:
            self._tmp_output_file = output_file
        args = (
            "nnenum.py",
            onnx_model_file.name,
            constraint_file.name,
            "-o",
            self._tmp_output_file.name,
        )
        if (
            "num_processes" in self.parameters
            and self.parameters["num_processes"] is not None
        ):
            value = self.parameters["num_processes"]
            args += (f"--num_processes={value}",)
        return args

    def parse_results(self, prop, results):
        result_str, cex = np.load(self._tmp_output_file.name, allow_pickle=True)
        if result_str == "safe":
            return UNSAT, None
        elif result_str == "unsafe":
            return SAT, cex
        raise self.translator_error(f"Unknown verification result: {result_str}")
