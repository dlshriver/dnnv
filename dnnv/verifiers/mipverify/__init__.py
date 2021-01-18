import numpy as np
import os
import subprocess as sp

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.results import SAT, UNSAT, UNKNOWN
from dnnv.verifiers.common.utils import as_layers

from .errors import MIPVerifyError, MIPVerifyTranslatorError
from .layers import MIPVERIFY_LAYER_TYPES
from .utils import to_mipverify_inputs


class MIPVerify(Verifier):
    translator_error = MIPVerifyTranslatorError
    verifier_error = MIPVerifyError

    @classmethod
    def is_installed(cls):
        verifier = "julia"
        for path in os.environ["PATH"].split(os.pathsep):
            exe = os.path.join(path, verifier)
            if os.path.isfile(exe) and os.access(exe, os.X_OK):
                proc = sp.run(
                    ["julia", "-e", "using MIPVerify"],
                    stdout=sp.DEVNULL,
                    stderr=sp.DEVNULL,
                )
                return proc.returncode == 0
        return False

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise MIPVerifyTranslatorError(
                "Unsupported network: More than 1 input variable"
            )
        if all((lb >= 0).all() for lb in prop.input_constraint.lower_bounds) and all(
            (ub <= 1).all() for ub in prop.input_constraint.upper_bounds
        ):
            lbs = prop.input_constraint.lower_bounds
            ubs = prop.input_constraint.upper_bounds
            op_graph = prop.suffixed_op_graph()
        else:
            op_graph, (lbs, ubs) = prop.prefixed_and_suffixed_op_graph()
        layers = as_layers(
            op_graph,
            extra_layer_types=MIPVERIFY_LAYER_TYPES,
            translator_error=MIPVerifyTranslatorError,
        )
        input_shape = prop.op_graph.input_shape[0]
        lb = lbs[0]
        ub = ubs[0]
        mipverify_inputs = to_mipverify_inputs(
            lb, ub, layers, translator_error=MIPVerifyTranslatorError,
        )
        return "julia", mipverify_inputs["property_path"]

    def parse_results(self, prop, results):
        stdout, stderr = results
        result = stdout[-1].lower()
        if "infeasible" in result:
            return UNSAT, None
        elif "optimal" in result:
            return SAT, None
        elif "trivial" in result:
            return SAT, None
        raise MIPVerifyTranslatorError(f"Unexpected verification result: {stdout[-1]}")
