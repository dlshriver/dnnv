import numpy as np

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.results import SAT, UNSAT, UNKNOWN
from dnnv.verifiers.common.utils import as_layers
from dnnv.verifiers.reluplex.utils import to_nnet_file

from .errors import MarabouError, MarabouTranslatorError
from .utils import get_marabou_properties

class Marabou(Verifier):
    translator_error = MarabouTranslatorError
    verifier_error = MarabouError

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise MarabouTranslatorError(
                "Unsupported network: More than 1 input variable"
            )
        layers = as_layers(
            prop.suffixed_op_graph(),
            translator_error=self.translator_error,
        )
        nnet_file_name = to_nnet_file(
            prop.input_constraint,
            layers,
            # dirname=dirname,
            translator_error=self.translator_error,
        )
        properties = get_marabou_properties()
        return "marabou", nnet_file_name, properties

    def parse_results(self, prop, results):
        stdout, stderr = results
        for line in stdout:
            if line.startswith("sat"):
                shape, dtype = prop.op_graph.input_details[0]
                cex = np.zeros(np.product(shape), dtype)
                found = False
                for line in stdout:
                    if found and line.startswith("x"):
                        index = int(line.split("x")[1].split(" =")[0])
                        cex[index] = float(line.split(" = ")[1])
                    if line.startswith("sat"):
                        found = True
                return SAT, cex.reshape(shape)
            elif line.startswith("unsat"):
                return UNSAT, None
        raise self.verifier_error(f"No verification result found")
