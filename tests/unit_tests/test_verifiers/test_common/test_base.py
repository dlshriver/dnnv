from functools import partial

import numpy as np
import pytest

from dnnv.nn import OperationGraph, operations
from dnnv.properties.expressions import *
from dnnv.verifiers.common.base import Parameter as VerifierParameter
from dnnv.verifiers.common.base import Verifier
from dnnv.verifiers.common.errors import VerifierError
from dnnv.verifiers.common.executors import VerifierExecutor
from dnnv.verifiers.common.reductions.iopolytope import (
    HalfspacePolytope,
    IOPolytopeReduction,
)
from dnnv.verifiers.common.results import SAT, UNSAT


class MockExecutor(VerifierExecutor):
    def run(self):
        return True


class MockVerifierUNSAT(Verifier):
    executor = MockExecutor
    reduction = partial(IOPolytopeReduction, HalfspacePolytope, HalfspacePolytope)

    def build_inputs(self, prop):
        return ("mock",)

    def parse_results(self, prop, results):
        return UNSAT, None


class MockVerifierSAT(Verifier):
    executor = MockExecutor
    parameters = {"cex": VerifierParameter(np.asarray)}
    reduction = partial(IOPolytopeReduction, HalfspacePolytope, HalfspacePolytope)

    def build_inputs(self, prop):
        return ("mock",)

    def parse_results(self, prop, results):
        return SAT, self.parameter_values["cex"]


def test_Verifier_init():
    x = Symbol("x")
    N = Network("N")
    expression = Forall(
        x, Implies(And(Constant(0) <= x, x <= Constant(1)), N(x) > Constant(0))
    )

    verifier = MockVerifierUNSAT(expression)
    assert len(verifier.parameter_values) == 0
    with pytest.raises(VerifierError):
        verifier = MockVerifierUNSAT(expression, extra_param=123)

    verifier = MockVerifierSAT(expression)
    assert len(verifier.parameter_values) == 1
    assert verifier.parameter_values["cex"] is None

    verifier = MockVerifierSAT(expression, cex=123)
    assert len(verifier.parameter_values) == 1
    assert verifier.parameter_values["cex"] == 123


def test_Verifier_verify_unsat():
    x = Symbol("x")
    N = Network("N")
    N.concretize(
        OperationGraph(
            [
                operations.Mul(
                    2.0,
                    operations.Relu(
                        operations.Mul(
                            2.0, operations.Input((-1, 1), np.dtype("float32"))
                        )
                    ),
                )
            ]
        )
    )
    expression = Forall(
        x, Implies(And(Constant(0) <= x, x <= Constant(1)), N(x) > Constant(0))
    )
    result = MockVerifierUNSAT.verify(expression)
    assert result[0] == UNSAT


def test_Verifier_verify_sat():
    x = Symbol("x")
    N = Network("N")
    N.concretize(
        OperationGraph(
            [
                operations.Mul(
                    2.0,
                    operations.Relu(
                        operations.Mul(
                            2.0, operations.Input((-1, 1), np.dtype("float32"))
                        )
                    ),
                )
            ]
        )
    )
    expression = Forall(
        x, Implies(And(Constant(0) <= x, x <= Constant(1)), N(x) < Constant(0))
    )
    result = MockVerifierSAT.verify(expression)
    assert result[0] == SAT

    result = MockVerifierSAT.verify(
        expression, cex=np.asarray([[0.5]], dtype=np.float32)
    )
    assert result[0] == SAT

    with pytest.raises(VerifierError):
        result = MockVerifierSAT.verify(
            expression, cex=np.asarray([[-1]], dtype=np.float32)
        )


def test_Verifier_verify_concrete():
    expression = Constant(False)
    result = MockVerifierSAT.verify(expression)
    assert result[0] == SAT
    assert result[1] is None

    expression = Constant(True)
    result = MockVerifierSAT.verify(expression)
    assert result[0] == UNSAT
    assert result[1] is None


def test_Verifier_verify_trivial():
    x = Symbol("x")
    x.ctx.shapes[x] = (1, 1)
    x.ctx.types[x] = np.dtype(np.float32)
    expression = Forall(
        x, Implies(And(Constant(0) <= x, x <= Constant(1)), Constant(True))
    )
    result = MockVerifierUNSAT.verify(expression)
    assert result[0] == UNSAT

    expression = Forall(
        x, Implies(And(Constant(0) <= x, x <= Constant(1)), Constant(False))
    )
    result = MockVerifierSAT.verify(expression)
    assert result[0] == SAT
    assert result[1] is not None


def test_Verifier_verify_trivial_with_hpoly():
    x = Symbol("x")
    x.ctx.shapes[x] = (1, 2)
    x.ctx.types[x] = np.dtype(np.float32)
    expression = Forall(
        x,
        Implies(
            And(Constant(0) <= x, x <= Constant(1), x[0, 0] < x[0, 1]), Constant(True)
        ),
    )
    result = MockVerifierUNSAT.verify(expression)
    assert result[0] == UNSAT

    expression = Forall(
        x,
        Implies(
            And(Constant(0) <= x, x <= Constant(1), x[0, 0] < x[0, 1]), Constant(False)
        ),
    )
    result = MockVerifierSAT.verify(expression)
    assert result[0] == SAT
    assert result[1] is not None
