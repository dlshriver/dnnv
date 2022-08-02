import numpy as np
import pytest

from dnnv.nn.utils import TensorDetails
from dnnv.properties.expressions import *
from dnnv.properties.transformers import LiftIfThenElse


def test_missing():
    class FakeExpression(Expression):
        pass

    with pytest.raises(
        ValueError, match="Unimplemented expression type: FakeExpression"
    ):
        LiftIfThenElse().generic_visit(FakeExpression())
    del FakeExpression


def test_Associative_no_ite():
    transformer = LiftIfThenElse()

    expr = Add(Symbol("a"), Constant(3))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Add)
    assert len(new_expr.expressions) == 2
    assert Symbol("a") in new_expr.expressions
    assert Constant(3) in new_expr.expressions


def test_Associative_single_ite():
    transformer = LiftIfThenElse()

    ite_expr = IfThenElse(Symbol("condition"), Symbol("T"), Symbol("F"))
    expr = Add(ite_expr, Constant(3))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, IfThenElse)
    assert new_expr.condition is Symbol("condition")
    assert isinstance(new_expr.t_expr, Add)
    assert Symbol("T") in new_expr.t_expr.expressions
    assert Constant(3) in new_expr.t_expr.expressions
    assert isinstance(new_expr.f_expr, Add)
    assert Symbol("F") in new_expr.f_expr.expressions
    assert Constant(3) in new_expr.f_expr.expressions


def test_Associative_multi_ite():
    transformer = LiftIfThenElse()

    ite_expr_1 = IfThenElse(Symbol("condition1"), Symbol("T1"), Symbol("F1"))
    ite_expr_2 = IfThenElse(Symbol("condition2"), Symbol("T2"), Symbol("F2"))
    expr = Add(ite_expr_1, Constant(3), ite_expr_2)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, IfThenElse)
    assert new_expr.condition is Symbol("condition1")
    assert isinstance(new_expr.t_expr, IfThenElse)
    assert new_expr.t_expr.condition is Symbol("condition2")
    assert isinstance(new_expr.t_expr.t_expr, Add)
    assert Symbol("T1") in new_expr.t_expr.t_expr.expressions
    assert Symbol("T2") in new_expr.t_expr.t_expr.expressions
    assert Constant(3) in new_expr.t_expr.t_expr.expressions
    assert isinstance(new_expr.t_expr.f_expr, Add)
    assert Symbol("T1") in new_expr.t_expr.f_expr.expressions
    assert Symbol("F2") in new_expr.t_expr.f_expr.expressions
    assert Constant(3) in new_expr.t_expr.f_expr.expressions
    assert isinstance(new_expr.f_expr, IfThenElse)
    assert new_expr.f_expr.condition is Symbol("condition2")
    assert isinstance(new_expr.f_expr.t_expr, Add)
    assert Symbol("F1") in new_expr.f_expr.t_expr.expressions
    assert Symbol("T2") in new_expr.f_expr.t_expr.expressions
    assert Constant(3) in new_expr.f_expr.t_expr.expressions
    assert isinstance(new_expr.f_expr.f_expr, Add)
    assert Symbol("F1") in new_expr.f_expr.f_expr.expressions
    assert Symbol("F2") in new_expr.f_expr.f_expr.expressions
    assert Constant(3) in new_expr.f_expr.f_expr.expressions


def test_Binary_no_ite():
    transformer = LiftIfThenElse()

    expr = Equal(Symbol("a"), Constant(3))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Equal)
    assert new_expr.expr1 is Symbol("a")
    assert new_expr.expr2 is Constant(3)


def test_Binary_single_ite():
    transformer = LiftIfThenElse()

    ite_expr = IfThenElse(Symbol("condition"), Symbol("T"), Symbol("F"))
    expr = Equal(ite_expr, Constant(3))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, IfThenElse)
    assert new_expr.condition is Symbol("condition")
    assert isinstance(new_expr.t_expr, Equal)
    assert new_expr.t_expr.expr1 is Symbol("T")
    assert new_expr.t_expr.expr2 is Constant(3)
    assert isinstance(new_expr.f_expr, Equal)
    assert new_expr.f_expr.expr1 is Symbol("F")
    assert new_expr.f_expr.expr2 is Constant(3)

    ite_expr = IfThenElse(Symbol("condition"), Symbol("T"), Symbol("F"))
    expr = NotEqual(Constant(3), ite_expr)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, IfThenElse)
    assert new_expr.condition is Symbol("condition")
    assert isinstance(new_expr.t_expr, NotEqual)
    assert new_expr.t_expr.expr1 is Constant(3)
    assert new_expr.t_expr.expr2 is Symbol("T")
    assert isinstance(new_expr.f_expr, NotEqual)
    assert new_expr.f_expr.expr1 is Constant(3)
    assert new_expr.f_expr.expr2 is Symbol("F")


def test_Binary_multi_ite():
    transformer = LiftIfThenElse()

    ite_expr_1 = IfThenElse(Symbol("condition1"), Symbol("T1"), Symbol("F1"))
    ite_expr_2 = IfThenElse(Symbol("condition2"), Symbol("T2"), Symbol("F2"))
    expr = Equal(ite_expr_1, ite_expr_2)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, IfThenElse)
    assert new_expr.condition is Symbol("condition1")
    assert isinstance(new_expr.t_expr, IfThenElse)
    assert new_expr.t_expr.condition is Symbol("condition2")
    assert isinstance(new_expr.t_expr.t_expr, Equal)
    assert new_expr.t_expr.t_expr.expr1 is Symbol("T1")
    assert new_expr.t_expr.t_expr.expr2 is Symbol("T2")
    assert isinstance(new_expr.t_expr.f_expr, Equal)
    assert new_expr.t_expr.t_expr.expr1 is Symbol("T1")
    assert new_expr.t_expr.f_expr.expr2 is Symbol("F2")
    assert isinstance(new_expr.f_expr, IfThenElse)
    assert new_expr.f_expr.condition is Symbol("condition2")
    assert isinstance(new_expr.f_expr.t_expr, Equal)
    assert new_expr.f_expr.t_expr.expr1 is Symbol("F1")
    assert new_expr.f_expr.t_expr.expr2 is Symbol("T2")
    assert isinstance(new_expr.f_expr.f_expr, Equal)
    assert new_expr.f_expr.t_expr.expr1 is Symbol("F1")
    assert new_expr.f_expr.f_expr.expr2 is Symbol("F2")


def test_Unary_no_ite():
    transformer = LiftIfThenElse()

    expr = Negation(Symbol("a"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Negation)
    assert new_expr.expr is Symbol("a")


def test_Unary_ite():
    transformer = LiftIfThenElse()

    ite_expr = IfThenElse(Symbol("condition"), Symbol("T"), Symbol("F"))
    expr = Negation(ite_expr)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, IfThenElse)
    assert new_expr.condition is Symbol("condition")
    assert isinstance(new_expr.t_expr, Negation)
    assert new_expr.t_expr.expr is Symbol("T")
    assert isinstance(new_expr.f_expr, Negation)
    assert new_expr.f_expr.expr is Symbol("F")


def test_samysweb_property_1():
    # based on property 1 of issue https://github.com/dlshriver/dnnv/issues/75

    N = Network("N")
    fake_network = lambda x: np.array([[0]])
    fake_network.input_details = (TensorDetails((1, 5), np.float32),)
    fake_network.input_shape = ((1, 5),)
    fake_network.output_details = (TensorDetails((1, 3), np.float32),)
    fake_network.output_shape = ((1, 3),)
    N.concretize(fake_network)

    x_ = Symbol("x_")
    x_min = np.array([[0, -200]])
    x_max = np.array([[100, 100]])
    expr = Constant(2.0) * IfThenElse(
        Constant(0.0)
        > IfThenElse(
            (Constant(0.1) * (x_[0, 1] - Constant(-210.0)))
            <= (Constant(-0.2) * (x_[0, 1] - Constant(100))),
            (Constant(0.1) * (x_[0, 1] - Constant(-210.0))),
            (Constant(-0.2) * (x_[0, 1] - Constant(100))),
        ),
        Constant(0.0),
        IfThenElse(
            (Constant(0.1) * (x_[0, 1] - Constant(-210.0)))
            <= (Constant(-0.2) * (x_[0, 1] - Constant(100))),
            (Constant(0.1) * (x_[0, 1] - Constant(-210.0))),
            (Constant(-0.2) * (x_[0, 1] - Constant(100))),
        ),
    )
    expected_expr = IfThenElse(
        IfThenElse(
            (Constant(0.1) * (x_[0, 1] - Constant(-210.0)))
            <= (Constant(-0.2) * (x_[0, 1] - Constant(100))),
            Constant(0.0) > (Constant(0.1) * (x_[0, 1] - Constant(-210.0))),
            Constant(0.0) > (Constant(-0.2) * (x_[0, 1] - Constant(100))),
        ),
        Constant(0.0),
        IfThenElse(
            (Constant(0.1) * (x_[0, 1] - Constant(-210.0)))
            <= (Constant(-0.2) * (x_[0, 1] - Constant(100))),
            (Constant(2.0) * Constant(0.1) * (x_[0, 1] - Constant(-210.0))),
            (Constant(2.0) * Constant(-0.2) * (x_[0, 1] - Constant(100))),
        ),
    ).propagate_constants()

    transformer = LiftIfThenElse()
    new_expr = transformer.visit(expr).propagate_constants()
    assert isinstance(new_expr, Expression)
    assert new_expr.is_equivalent(expected_expr)
