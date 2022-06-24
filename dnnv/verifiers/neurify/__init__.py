from functools import partial

import numpy as np

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.reductions import HalfspacePolytope, IOPolytopeReduction
from dnnv.verifiers.common.results import SAT, UNKNOWN, UNSAT
from dnnv.verifiers.common.utils import as_layers

from .errors import NeurifyError, NeurifyTranslatorError
from .utils import to_neurify_inputs


class Neurify(Verifier):
    reduction = partial(IOPolytopeReduction, HalfspacePolytope, HalfspacePolytope)
    translator_error = NeurifyTranslatorError
    verifier_error = NeurifyError
    parameters = {
        "max_depth": Parameter(int, help="Maximum search depth for neurify."),
        "max_thread": Parameter(int, help="Maximum number of threads to use."),
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
        neurify_inputs = to_neurify_inputs(
            prop.input_constraint,
            layers,
            # dirname=dirname,
            translator_error=self.translator_error,
        )
        return (
            "neurify",
            "-n",
            neurify_inputs["nnet_path"],
            "-x",
            neurify_inputs["input_path"],
            "-sl",
            "0.000000000001",  # TODO: remove magic number
            "-I",
            neurify_inputs["input_interval_path"],
            "-H",
            neurify_inputs["input_hpoly_path"],
            "-v",
        ) + tuple(
            f"--{k}={v}" for k, v in self.parameter_values.items() if v is not None
        )

    def parse_results(self, prop, results):
        stdout, stderr = results
        if len(stdout) < 2:
            raise self.verifier_error(
                f"Verifier terminated before producing expected output."
            )
        result = stdout[-2].strip()
        if result == "Falsified.":
            cex_found = False
            input_shape, input_dtype = prop.op_graph.input_details[0]
            for line in stdout:
                if cex_found:
                    values = line.split(":")[-1][1:-1].split()
                    cex = np.asarray(
                        [float(v) for v in values], dtype=input_dtype
                    ).reshape(input_shape)
                    break
                if line.endswith("Solution:"):
                    cex_found = True
            else:
                # counter example was found in first pass, not printed
                lb, ub = prop.input_constraint.as_bounds()
                lb = lb.reshape(input_shape)
                ub = ub.reshape(input_shape)
                cex = ((lb + ub) / 2).astype(input_dtype)
            return SAT, cex
        elif result == "Unknown.":
            return UNKNOWN, None
        elif result == "Proved.":
            return UNSAT, None
        raise self.verifier_error(f"Unexpected verification result: {stdout[-1]}")
