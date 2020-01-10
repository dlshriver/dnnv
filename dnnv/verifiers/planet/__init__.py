import tempfile

from typing import List, Optional, Type

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

from .errors import PlanetError, PlanetTranslatorError
from .layers import PLANET_LAYER_TYPES
from .utils import to_rlv_file


def parse_results(stdout: List[str], stderr: List[str]):
    if stdout[-1] == "SAT":
        return SAT
    elif stdout[-1] == "UNSAT":
        return UNSAT
    raise PlanetError(f"Unexpected verification result: {stdout[-1]}")


def verify(dnn: OperationGraph, phi: Expression):
    dnn = dnn.simplify()
    phi.networks[0].concretize(dnn)

    result = UNSAT
    property_extractor = ConvexPolytopeExtractor()
    with tempfile.TemporaryDirectory() as dirname:
        for prop in property_extractor.extract_from(phi):
            layers = prop.output_constraint.as_layers(
                prop.network,
                extra_layer_types=PLANET_LAYER_TYPES,
                translator_error=PlanetTranslatorError,
            )
            input_interval = prop.input_constraint.as_hyperrectangle()
            rlv_file_name = to_rlv_file(
                input_interval,
                layers,
                dirname=dirname,
                translator_error=PlanetTranslatorError,
            )
            executor = CommandLineExecutor(
                "planet", f"{rlv_file_name}", verifier_error=PlanetError
            )
            out, err = executor.run()
            result |= parse_results(out, err)
            if result == SAT:
                # TODO : remove
                print("DEBUGGING (checking counter example)")
                import numpy as np

                shape, dtype = prop.network.value.input_details[0]
                solution = np.zeros(shape, dtype)
                found = False
                for line in out:
                    if line.startswith("SAT"):
                        found = True
                    if found and line.startswith("- input"):
                        position = tuple(int(i) for i in line.split(":")[1:-1])
                        value = float(line.split()[-1])
                        solution[position] = value
                if prop.network.value.output_operations[0].__class__.__name__ == "Relu":
                    output = prop.network.value(solution)
                else:
                    output = prop.network.value[:-2](solution)
                print("DEBUGGING (original network output)", output)
                output = output @ layers[-2].weights + layers[-2].bias
                print(
                    "DEBUGGING (constraint outputs (zero means SAT))",
                    np.clip(output, 0.0, None),
                )
                print(
                    "DEBUGGING (final result (zero means SAT))",
                    np.clip(output, 0.0, None) @ layers[-1].weights + layers[-1].bias,
                )
            if result == SAT or result == UNKNOWN:
                return result

    return result
