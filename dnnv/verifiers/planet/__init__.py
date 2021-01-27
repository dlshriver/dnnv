import numpy as np

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.reductions import IOPolytopeReduction, HalfspacePolytope
from dnnv.verifiers.common.results import SAT, UNSAT, UNKNOWN
from dnnv.verifiers.common.utils import as_layers
from functools import partial

from .errors import PlanetError, PlanetTranslatorError
from .layers import PLANET_LAYER_TYPES
from .utils import to_rlv_file


class Planet(Verifier):
    reduction = partial(IOPolytopeReduction, HalfspacePolytope, HalfspacePolytope)
    translator_error = PlanetTranslatorError
    verifier_error = PlanetError

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise ReluplexTranslatorError(
                "Unsupported network: More than 1 input variable"
            )
        layers = as_layers(
            prop.suffixed_op_graph(),
            extra_layer_types=PLANET_LAYER_TYPES,
            translator_error=self.translator_error,
        )
        rlv_file_name = to_rlv_file(
            prop.input_constraint,
            layers,
            # dirname=dirname,
            translator_error=self.translator_error,
        )
        return "planet", rlv_file_name

    def parse_results(self, prop, results):
        stdout, stderr = results
        if len(stdout) == 0:
            raise self.verifier_error(f"Running planet produced no output.")
        if stdout[-1] == "SAT":
            shape, dtype = prop.op_graph.input_details[0]
            cex = np.zeros((1,) + shape[1:], dtype)
            found = False
            for line in stdout:
                if line.startswith("SAT"):
                    found = True
                if found and line.startswith("- input"):
                    position = tuple(int(i) for i in line.split(":")[1:-1])
                    value = float(line.split()[-1])
                    cex[position] = value
            return SAT, cex.reshape(shape)
        elif stdout[-1] == "UNSAT":
            return UNSAT, None
        raise self.verifier_error(f"Unexpected verification result: {stdout[-1]}")

    def validate_counter_example(self, prop, cex):
        # planet output has a precision of 5 decimal places
        is_valid, err_msg = prop.validate_counter_example(cex, threshold=1e-5)
        if not is_valid:
            raise self.verifier_error(err_msg)
        return is_valid
