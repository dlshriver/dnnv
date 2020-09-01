import numpy as np

from dnnv.verifiers.common.base import Parameter, Verifier
from dnnv.verifiers.common.executors import VerifierExecutor
from dnnv.verifiers.common.results import SAT, UNSAT, UNKNOWN
from dnnv.verifiers.common.utils import as_layers
from dnnv.verifiers.planet.utils import to_rlv_file

from .errors import BabError, BabTranslatorError


class BaBExecutor(VerifierExecutor):
    def run(self):
        from plnn.branch_and_bound import bab
        from plnn.dual_network_linear_approximation import LooseDualNetworkApproximation
        from plnn.model import load_and_simplify
        from plnn.network_linear_approximation import LinearizedNetwork

        input_path, reluify_maxpools, smart_branching = self.args

        with open(input_path) as input_file:
            network, domain = load_and_simplify(input_file, LinearizedNetwork)

        if reluify_maxpools:
            network.remove_maxpools(domain)

        smart_brancher = None
        if smart_branching:
            with open(input_path) as input_file:
                smart_brancher, _ = load_and_simplify(
                    input_file, LooseDualNetworkApproximation
                )
            smart_brancher.remove_maxpools(domain)

        epsilon = 1e-2
        decision_bound = 0
        min_lb, min_ub, ub_point, nb_visited_states = bab(
            network, domain, epsilon, decision_bound, smart_brancher
        )
        return min_lb, min_ub, ub_point, nb_visited_states


class BaB(Verifier):
    translator_error = BabTranslatorError
    verifier_error = BabError
    parameters = {
        "reluify_maxpools": Parameter(bool, default=False),
        "smart_branching": Parameter(bool, default=False),
    }
    executor = BaBExecutor

    @classmethod
    def is_installed(cls):
        try:
            from plnn.branch_and_bound import bab
            from plnn.dual_network_linear_approximation import (
                LooseDualNetworkApproximation,
            )
            from plnn.model import load_and_simplify
            from plnn.network_linear_approximation import LinearizedNetwork
        except ImportError:
            return False
        return True

    def build_inputs(self, prop):
        if prop.input_constraint.num_variables > 1:
            raise self.translator_error(
                "Unsupported network: More than 1 input variable"
            )
        layers = as_layers(
            prop.suffixed_op_graph(), translator_error=self.translator_error,
        )
        rlv_file_path = to_rlv_file(
            prop.input_constraint,
            layers,
            # dirname=dirname,
            translator_error=self.translator_error,
        )
        return (
            rlv_file_path,
            self.parameters["reluify_maxpools"],
            self.parameters["smart_branching"],
        )

    def parse_results(self, prop, results):
        min_lb, min_ub, ub_point, nb_visited_states = results
        if min_lb > 0:
            return UNSAT, None
        elif min_ub <= 0:
            candidate_ctx = ub_point.view(1, -1)
            val = network.net(candidate_ctx)
            margin = val.squeeze().item()
            if margin > 0:
                raise self.verifier_error("Invalid counter example found")
            return SAT, candidate_ctx.cpu().detach().numpy()
        else:
            return UNKNOWN, None


# def parse_args(args):
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--bab.reluify_maxpools", action="store_true", dest="reluify_maxpools"
#     )
#     parser.add_argument(
#         "--bab.smart_branching", action="store_true", dest="smart_branching"
#     )
#     return parser.parse_known_args(args)


# def check(input_path, reluify_maxpools=False, smart_branching=False):
#     with open(input_path) as input_file:
#         network, domain = load_and_simplify(input_file, LinearizedNetwork)

#     if reluify_maxpools:
#         network.remove_maxpools(domain)

#     smart_brancher = None
#     if smart_branching:
#         with open(input_path) as input_file:
#             smart_brancher, _ = load_and_simplify(
#                 input_file, LooseDualNetworkApproximation
#             )
#         smart_brancher.remove_maxpools(domain)

#     epsilon = 1e-2
#     decision_bound = 0
#     min_lb, min_ub, ub_point, nb_visited_states = bab(
#         network, domain, epsilon, decision_bound, smart_brancher
#     )
#     if min_lb > 0:
#         return UNSAT
#     elif min_ub <= 0:
#         candidate_ctx = ub_point.view(1, -1)
#         val = network.net(candidate_ctx)
#         margin = val.squeeze().item()
#         if margin > 0:
#             raise BabError("Invalid counter example found")
#         return SAT
#     else:
#         return UNKNOWN


# def verify(dnn: OperationGraph, phi: Expression, **kwargs):
#     dnn = dnn.simplify()
#     phi.networks[0].concretize(dnn)

#     result = UNSAT
#     property_extractor = HalfspacePolytopePropertyExtractor(
#         HyperRectangle, HalfspacePolytope
#     )
#     with tempfile.TemporaryDirectory() as dirname:
#         for prop in property_extractor.extract_from(~phi):
#             if prop.input_constraint.num_variables > 1:
#                 raise BabTranslatorError(
#                     "Unsupported network: More than 1 input variable"
#                 )
#             layers = as_layers(
#                 prop.suffixed_op_graph(), translator_error=BabTranslatorError,
#             )
#             rlv_file_path = to_rlv_file(
#                 prop.input_constraint,
#                 layers,
#                 dirname=dirname,
#                 translator_error=BabTranslatorError,
#             )
#             result |= check(rlv_file_path, **kwargs)
#             if result == SAT or result == UNKNOWN:
#                 return result

#     return result
