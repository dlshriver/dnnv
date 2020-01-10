import argparse
import numpy as np
import tensorflow.compat.v1 as tf

from eran import ERAN

from dnnv import logging
from dnnv.nn.layers import Convolutional, FullyConnected, InputLayer
from dnnv.verifiers.common import SAT, UNSAT, UNKNOWN, PropertyExtractor, as_layers

from .errors import ERANTranslatorError
from .layers import ERAN_LAYER_TYPES, conv_as_tf


class ERANTranslator:
    def __init__(self, dnn, phi):
        dnn = dnn.simplify()
        networks = phi.networks
        if len(networks) == 0:
            raise ERANTranslatorError("Property does not use a network")
        if len(networks) > 1:
            raise ERANTranslatorError("Property has more than 1 network")
        network = networks[0]
        network.concretize(dnn)
        self.phi = phi.propagate_constants().to_cnf()
        self.not_phi = ~self.phi
        self.layers = []
        self.property_checks = {}

    def __iter__(self):
        property_extractor = PropertyExtractor()
        for conjunction in self.not_phi:
            op_graph, constraint_type, (
                lower_bound,
                upper_bound,
            ), output_constraint = property_extractor.extract(conjunction)
            input_bounds = (tuple(lower_bound.flatten()), tuple(upper_bound.flatten()))
            property_check = (constraint_type, input_bounds, op_graph)
            if property_check not in self.property_checks:
                self.property_checks[property_check] = {
                    "input_bounds": (lower_bound, upper_bound),
                    "output_constraints": [],
                }
            self.property_checks[property_check]["output_constraints"].append(
                output_constraint
            )
        for (
            (constraint_type, _, op_graph),
            constraint,
        ) in self.property_checks.items():
            self.layers = as_layers(
                op_graph,
                layer_types=[InputLayer, FullyConnected, Convolutional]
                + ERAN_LAYER_TYPES,
            )
            input_bounds = constraint["input_bounds"]
            output_constraints = constraint["output_constraints"]
            if constraint_type == "regression":
                pass
            elif constraint_type == f"classification-argmax":
                pass
            else:
                raise ERANTranslatorError(
                    f"Unsupported property type for ERAN: {constraint_type}"
                )
            with tf.Session(graph=tf.Graph()) as sess:
                yield ERANCheck(self.as_tf(), input_bounds, output_constraints, sess)

    def as_tf(self, include_input=False):
        input_size = np.asarray(self.layers[0].shape)
        if len(input_size) == 4:
            input_size = input_size[[0, 2, 3, 1]]
        input_size = [d if d >= 0 else None for d in input_size]
        input_placeholder = x = tf.placeholder(tf.float32, input_size)
        for layer in self.layers[1:]:
            if isinstance(layer, FullyConnected):
                if len(x.shape) != 2:
                    x = tf.reshape(x, (x.shape[0], -1))
                x = tf.nn.bias_add(tf.matmul(x, layer.weights), layer.bias)
                if layer.activation == "relu":
                    x = tf.nn.relu(x)
                elif layer.activation is not None:
                    raise ERANTranslatorError(
                        f"{layer.activation} activation is currently unsupported"
                    )
            elif isinstance(layer, Convolutional):
                x = conv_as_tf(layer, x)
            elif isinstance(layer, tuple(ERAN_LAYER_TYPES)):
                x = layer.as_tf(x)
            else:
                raise ERANTranslatorError(
                    f"Unsupported layer type for ERAN: {type(layer).__name__}"
                )
        if include_input:
            return input_placeholder, x
        return x


class ERANCheck:
    def __init__(self, network, input_bounds, output_constraint, session):
        self.network = network
        self.input_lower_bound = np.asarray(input_bounds[0])
        self.input_upper_bound = np.asarray(input_bounds[1])
        if self.input_lower_bound.ndim == 4:
            self.input_lower_bound = self.input_lower_bound.transpose((0, 2, 3, 1))
        if self.input_upper_bound.ndim == 4:
            self.input_upper_bound = self.input_upper_bound.transpose((0, 2, 3, 1))
        self.output_constraint = output_constraint
        self.session = session

    def check_constraint(
        self, constraint_type, value, output_lower_bound, output_upper_bound
    ):
        if constraint_type == ">":
            if output_upper_bound <= value:
                return UNSAT
            elif output_lower_bound <= value:
                return UNKNOWN
        elif constraint_type == ">=":
            if output_upper_bound < value:
                return UNSAT
            elif output_lower_bound < value:
                return UNKNOWN
        elif constraint_type == "<":
            if output_lower_bound >= value:
                return UNSAT
            elif output_upper_bound >= value:
                return UNKNOWN
        elif constraint_type == "<=":
            if output_lower_bound > value:
                return UNSAT
            elif output_upper_bound > value:
                return UNKNOWN
        elif constraint_type == "!=":
            partial_result = SAT
            # check ==, then invert
            for i, (vlb, vub) in enumerate(zip(output_lower_bound, output_upper_bound)):
                if i == value:
                    continue
                if output_lower_bound[value] < vub:
                    partial_result &= UNKNOWN
                if output_upper_bound[value] < vlb:
                    partial_result &= UNSAT
            return ~partial_result
        else:
            raise ERANTranslatorError(f"Unsupported constraint type: {constraint_type}")
        return SAT

    def check(
        self, domain="deeppoly", timeout_lp=1, timeout_milp=1, use_area_heuristic=True
    ):
        logger = logging.getLogger(__name__)
        spec_lb = self.input_lower_bound.flatten().copy()
        spec_ub = self.input_upper_bound.flatten().copy()
        eran_model = ERAN(self.network, session=self.session)
        _, nn, nlb, nub = eran_model.analyze_box(
            spec_lb,
            spec_ub,
            domain=domain,
            timeout_lp=timeout_lp,
            timeout_milp=timeout_milp,
            use_area_heuristic=use_area_heuristic,
        )
        output_lower_bound = np.asarray(nlb[-1])
        output_upper_bound = np.asarray(nub[-1])
        logger.info("output lower bound: %s", output_lower_bound)
        logger.info("output upper bound: %s", output_upper_bound)

        result = UNSAT
        for constraint in self.output_constraint:
            partial_result = SAT
            for constraint_type, value in constraint.items():
                partial_result &= self.check_constraint(
                    constraint_type, value, output_lower_bound, output_upper_bound
                )
            if partial_result == SAT:
                return SAT
            result |= partial_result
        return result


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eran.domain",
        default="deeppoly",
        choices=["deepzono", "deeppoly", "refinezono", "refinepoly"],
        dest="domain",
    )
    parser.add_argument("--eran.timeout_lp", default=1.0, type=float, dest="timeout_lp")
    parser.add_argument(
        "--eran.timeout_milp", default=1.0, type=float, dest="timeout_milp"
    )
    parser.add_argument(
        "--eran.dont_use_area_heuristic",
        action="store_false",
        dest="use_area_heuristic",
    )
    return parser.parse_known_args(args)


def verify(dnn, phi, **kwargs):
    eran_translator = ERANTranslator(dnn, phi)
    result = UNSAT
    for eran_property in eran_translator:
        result |= eran_property.check(**kwargs)
    return result
