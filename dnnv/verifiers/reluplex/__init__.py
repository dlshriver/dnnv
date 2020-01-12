import tempfile

from typing import List

from dnnv import logging
from dnnv.nn import OperationGraph
from dnnv.properties import Expression
from dnnv.verifiers.common import (
    SAT,
    UNSAT,
    UNKNOWN,
    CommandLineExecutor,
    ConvexPolytopeExtractor,
)

from .errors import ReluplexError, ReluplexTranslatorError
from .utils import to_nnet_file


def parse_results(stdout: List[str], stderr: List[str]):
    for line in stdout:
        if line.startswith("Solution found!"):
            return SAT
        elif line.startswith("Can't solve!"):
            return UNSAT
    raise ReluplexError(f"No verification result found")


def verify(dnn: OperationGraph, phi: Expression):
    dnn = dnn.simplify()
    phi.networks[0].concretize(dnn)

    result = UNSAT
    property_extractor = ConvexPolytopeExtractor()
    with tempfile.TemporaryDirectory() as dirname:
        for prop in property_extractor.extract_from(phi):
            layers = prop.output_constraint.as_layers(
                prop.network, translator_error=ReluplexTranslatorError
            )
            input_interval = prop.input_constraint.as_hyperrectangle()
            nnet_file_name = to_nnet_file(
                input_interval,
                layers,
                dirname=dirname,
                translator_error=ReluplexTranslatorError,
            )
            executor = CommandLineExecutor(
                "reluplex", f"{nnet_file_name}", verifier_error=ReluplexError
            )
            out, err = executor.run()
            result |= parse_results(out, err)
            # TODO : double check whether counter example is valid
            # if result == SAT:
            #     import numpy as np

            #     input_shape, input_type = prop.network.value.input_details[0]
            #     solution = np.zeros(np.product(input_shape), input_type)
            #     found = False
            #     for line in out:
            #         if found and line.startswith("input"):
            #             index = int(line.split("]", maxsplit=1)[0].split("[")[-1])
            #             solution[index] = float(line.split()[-1][:-1])
            #         if line.startswith("Solution found!"):
            #             found = True
            #     print(prop.network.value(solution.reshape(input_shape))) # TODO : remove
            if result == SAT or result == UNKNOWN:
                return result

    return result
