import numpy as np

from dnnv.nn.utils import TensorDetails
from dnnv.properties import *
from dnnv.properties.transformers import CanonicalTransformer


def test_Multiply_symbols():
    transformer = CanonicalTransformer()

    expr = Multiply(Symbol("a"), Symbol("b"), Symbol("c"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert len(new_expr.expressions) == 1
    assert (
        Multiply(Constant(1), Symbol("a"), Symbol("b"), Symbol("c"))
        in new_expr.expressions
    )


def test_Multiply_constants():
    transformer = CanonicalTransformer()

    expr = Multiply(Constant(1), Constant(-3))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(-3)


def test_Multiply_mixed_constants_symbols():
    transformer = CanonicalTransformer()

    expr = Multiply(Constant(1), Symbol("a"), Constant(-3), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert len(new_expr.expressions) == 1
    assert Multiply(Constant(-3), Symbol("a"), Symbol("b")) in new_expr.expressions


def test_Multiply_mixed_additions():
    transformer = CanonicalTransformer()

    expr = Multiply(
        Add(Constant(1), Symbol("a")),
        Add(Constant(-3), Symbol("b"), Symbol("a")),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert len(new_expr.expressions) == 5
    assert Multiply(Constant(1), Constant(-3)) in new_expr
    assert Multiply(Constant(-2), Symbol("a")) in new_expr
    assert Multiply(Constant(1), Symbol("b")) in new_expr
    assert Multiply(Constant(1), Symbol("a"), Symbol("b")) in new_expr
    assert Multiply(Constant(1), Symbol("a"), Symbol("a")) in new_expr


def test_samysweb_property_3():
    # based on property 3 of issue https://github.com/dlshriver/dnnv/issues/75
    transformer = CanonicalTransformer()

    x_ = Symbol("x_")
    N = Network("N")
    fake_network = lambda x: np.array([[0]])
    fake_network.input_details = (TensorDetails((1, 2), np.float32),)
    fake_network.input_shape = ((1, 2),)
    fake_network.output_details = (TensorDetails((1, 1), np.float32),)
    fake_network.output_shape = ((1, 1),)
    N.concretize(fake_network)

    expr = Not(
        Or(
            (
                Constant(0.0)
                >= IfThenElse(
                    Constant(0.0) > x_[0, 1],
                    (Constant(0.16) * x_[0, 0] + Constant(-0.16) * x_[0, 0]),
                    x_[0, 1],
                )
            ),
            And(x_[0, 1] == Constant(0.0), N(x_) == Constant(0.0)),
        )
    )

    new_expr = transformer.visit(expr).propagate_constants()
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        Or(
            And(
                LessThan(Multiply(-1, Network("N")(Symbol("x_"))), Constant(0)),
                LessThan(Multiply(-1, Symbol("x_")[(0, 1)]), Constant(0)),
                LessThanOrEqual(Multiply(-1, Symbol("x_")[(0, 1)]), Constant(0)),
            ),
            And(
                LessThan(Multiply(-1, Symbol("x_")[(0, 1)]), Constant(0)),
                LessThan(Network("N")(Symbol("x_")), Constant(0)),
                LessThanOrEqual(Multiply(-1, Symbol("x_")[(0, 1)]), Constant(0)),
            ),
            And(
                LessThan(Multiply(-1, Symbol("x_")[(0, 1)]), Constant(0)),
                LessThan(Symbol("x_")[(0, 1)], Constant(0)),
                LessThanOrEqual(Multiply(-1, Symbol("x_")[(0, 1)]), Constant(0)),
            ),
            And(
                LessThan(Multiply(-1, Symbol("x_")[(0, 1)]), Constant(0)),
                LessThanOrEqual(Multiply(-1, Symbol("x_")[(0, 1)]), Constant(0)),
            ),
        )
    )
