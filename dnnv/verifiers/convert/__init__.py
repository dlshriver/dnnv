import shutil

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.executors import VerifierExecutor
from dnnv.verifiers.common.reductions import (
    IOPolytopeReduction,
    HalfspacePolytope,
    HyperRectangle,
)
from dnnv.verifiers.common.results import UNKNOWN
from dnnv.verifiers.common.utils import as_layers
from dnnv.verifiers.planet.errors import PlanetTranslatorError
from dnnv.verifiers.planet.utils import to_rlv_file
from dnnv.verifiers.reluplex.errors import ReluplexTranslatorError
from dnnv.verifiers.reluplex.utils import to_nnet_file
from functools import partial
from pathlib import Path

from .errors import VnnlibTranslatorError
from .utils import to_vnnlib_onnx_file, to_vnnlib_property_file


class DummyExecutor(VerifierExecutor):
    def run(self):
        return


class Convert(Verifier):
    translator_error = VnnlibTranslatorError
    parameters = {
        "to": Parameter(str, default="vnnlib", choices=["vnnlib", "rlv", "nnet"],),
        "dest": Parameter(Path, default=Path(".")),
    }
    executor = DummyExecutor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_format = self.parameters.get("to")
        if output_format in ["rlv", "nnet"]:
            self.__class__.reduction = partial(
                IOPolytopeReduction, HyperRectangle, HalfspacePolytope
            )
        else:
            self.__class__.reduction = partial(
                IOPolytopeReduction, HalfspacePolytope, HalfspacePolytope
            )
        self.property_id = 0

    @classmethod
    def is_installed(cls):
        return True

    def run(self):
        super().run()
        exit(0)

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise self.translator_error(
                "Unsupported network: More than 1 input variable"
            )
        paths = []
        if self.parameters.get("to") == "nnet":
            layers = as_layers(
                prop.suffixed_op_graph(), translator_error=self.translator_error,
            )
            paths.append(
                Path(
                    to_nnet_file(
                        prop.input_constraint,
                        layers,
                        translator_error=ReluplexTranslatorError,
                    )
                )
            )
        elif self.parameters.get("to") == "rlv":
            layers = as_layers(
                prop.suffixed_op_graph(), translator_error=self.translator_error,
            )
            paths.append(
                Path(
                    to_rlv_file(
                        prop.input_constraint,
                        layers,
                        translator_error=PlanetTranslatorError,
                    )
                )
            )
        elif self.parameters.get("to") == "vnnlib":
            paths.append(
                Path(
                    to_vnnlib_onnx_file(
                        prop.op_graph, translator_error=self.translator_error
                    )
                )
            )
            paths.append(
                Path(
                    to_vnnlib_property_file(
                        prop, translator_error=self.translator_error
                    )
                )
            )
            # TODO : create ONNX model
        dest: Path = self.parameters.get("dest")
        dest.mkdir(exist_ok=True, parents=True)
        for path in paths:
            suffix = "".join(path.suffixes)
            shutil.copy(path, dest / f"property{self.property_id}{suffix}")
        self.property_id += 1
        return ()

    def parse_results(self, prop, results):
        return UNKNOWN, None
