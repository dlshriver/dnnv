import numpy as np

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
                return True
        return False

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise MIPVerifyTranslatorError(
                "Unsupported network: More than 1 input variable"
            )
        layers = as_layers(
            prop.suffixed_op_graph(),
            extra_layer_types=MIPVERIFY_LAYER_TYPES,
            translator_error=MIPVerifyTranslatorError,
        )
        mipverify_inputs = to_mipverify_inputs(
            prop.input_constraint,
            layers,
            # dirname=dirname,
            translator_error=MIPVerifyTranslatorError,
        )
        return "julia", mipverify_inputs["property_path"]

    def parse_results(self, prop, results):
        result = stdout[-1].lower()
        if "infeasible" in result:
            return UNSAT, None
        elif "optimal" in result:
            return SAT, None
        raise MIPVerifyTranslatorError(f"Unexpected verification result: {stdout[-1]}")
