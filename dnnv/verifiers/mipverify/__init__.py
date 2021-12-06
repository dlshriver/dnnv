import h5py
import numpy as np

from dnnv.nn import operations
from dnnv.nn.graph import OperationGraph
from dnnv.verifiers.common.base import Verifier
from dnnv.verifiers.common.results import SAT, UNSAT, UNKNOWN

from .errors import MIPVerifyError, MIPVerifyTranslatorError
from .utils import to_mipverify_inputs


class MIPVerify(Verifier):
    translator_error = MIPVerifyTranslatorError
    verifier_error = MIPVerifyError

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise self.translator_error(
                "Unsupported network: More than 1 input variable"
            )
        lbs = prop.input_constraint.lower_bounds[0]
        ubs = prop.input_constraint.upper_bounds[0]
        if (lbs >= 0).all() and (ubs <= 1).all():
            op_graph = prop.suffixed_op_graph()
            self._tmp_input_denormalizer = lambda x: x
        else:
            (
                op_graph,
                (new_lbs, new_ubs),
                input_denormalizers,
            ) = prop.prefixed_and_suffixed_op_graph(return_prefixes=True)
            lbs = new_lbs[0]
            ubs = new_ubs[0]
            self._tmp_input_denormalizer = input_denormalizers[0]
        output_dtype = op_graph.output_details[0].dtype
        two_class_output = operations.Gemm(
            op_graph.output_operations[0],
            np.array([[-1.0, 1.0]], dtype=output_dtype),
        )
        robustness_op_graph = OperationGraph([two_class_output]).simplify()

        mipverify_inputs = to_mipverify_inputs(
            lbs,
            ubs,
            robustness_op_graph,
            translator_error=self.translator_error,
        )
        self._tmp_output_file = mipverify_inputs["cex_file_path"]
        return "mipverify", mipverify_inputs["julia_file_path"]

    def parse_results(self, prop, results):
        stdout, _ = results
        result = stdout[-1].strip().lower()
        if "infeasible" in result:
            return UNSAT, None
        elif result == "optimal" or result == "trivial":
            with h5py.File(self._tmp_output_file, "r") as f:
                cex_ds = f["cex"]
                cex_ = cex = np.asarray(cex_ds).reshape(cex_ds.shape[::-1])
            if cex.ndim == 4:
                cex = cex.transpose((0, 3, 2, 1))
            dtype = prop.op_graph.input_details[0].dtype
            cex = cex.astype(dtype)

            cex = self._tmp_input_denormalizer(cex)
            lbs = prop.input_constraint.lower_bounds[0]
            ubs = prop.input_constraint.upper_bounds[0]
            if np.any(cex < lbs) or np.any(cex > ubs):
                return UNKNOWN, None

            return SAT, cex
        raise self.translator_error(f"Unexpected verification result: {result}")
