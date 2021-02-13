import numpy as np

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.results import SAT, UNSAT, UNKNOWN
from dnnv.verifiers.common.utils import as_layers

from .errors import ReluplexError, ReluplexTranslatorError
from .utils import to_nnet_file


class Reluplex(Verifier):
    translator_error = ReluplexTranslatorError
    verifier_error = ReluplexError

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise ReluplexTranslatorError(
                "Unsupported network: More than 1 input variable"
            )
        layers = as_layers(
            prop.suffixed_op_graph(), translator_error=self.translator_error,
        )
        nnet_file_name = to_nnet_file(
            prop.input_constraint,
            layers,
            # dirname=dirname,
            translator_error=self.translator_error,
        )
        return "reluplex", nnet_file_name

    def parse_results(self, prop, results):
        stdout, stderr = results
        for line in stdout:
            if line.startswith("Solution found!"):
                shape, dtype = prop.op_graph.input_details[0]
                cex = np.zeros(np.product([d if d > 0 else 1 for d in shape]), dtype)
                found = False
                for line in stdout:
                    if found and line.startswith("input"):
                        index = int(line.split("]", maxsplit=1)[0].split("[")[-1])
                        cex[index] = float(line.split()[-1][:-1])
                    if line.startswith("Solution found!"):
                        found = True
                return SAT, cex.reshape(shape)
            elif line.startswith("Can't solve!"):
                return UNSAT, None
        raise self.verifier_error(f"No verification result found")
