import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import GenericExpressionTransformer


def test_missing():
    class FakeExpression(Expression):
        pass

    with pytest.raises(
        ValueError, match="Unimplemented expression type: FakeExpression"
    ):
        GenericExpressionTransformer().visit(FakeExpression())
    del FakeExpression


def test_non_expression():
    transformer = GenericExpressionTransformer()
    new_expr = transformer.visit("non expression value")
    assert new_expr is Constant("non expression value")


def test_AssociativeExpression():
    transformer = GenericExpressionTransformer()

    expr = Add(Symbol("a"), Symbol("b"), Symbol("c"), Symbol("d"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert repr(new_expr) == repr(expr)


def test_BinaryExpression():
    transformer = GenericExpressionTransformer()

    expr = Equal(Symbol("a"), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert repr(new_expr) == repr(expr)


def test_Call():
    transformer = GenericExpressionTransformer()

    expr = Call(Symbol("f"), (), {})
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert repr(new_expr) == repr(expr)


def test_Constant():
    transformer = GenericExpressionTransformer()

    expr = Constant(3)
    new_expr = transformer.visit(expr)
    assert new_expr is expr


def test_Image(tmp_path):
    transformer = GenericExpressionTransformer()

    expr = Image(tmp_path / "test.npy")
    new_expr = transformer.visit(expr)
    assert new_expr is expr
    assert repr(new_expr) == repr(expr)

    expr = Image(Constant(tmp_path / "test.npy"))
    new_expr = transformer.visit(expr)
    assert new_expr is expr
    assert repr(new_expr) == repr(expr)


def test_TernaryExpression():
    transformer = GenericExpressionTransformer()

    expr = IfThenElse(Symbol("a"), Symbol("b"), Symbol("c"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert repr(new_expr) == repr(expr)


def test_Quantifier():
    transformer = GenericExpressionTransformer()

    expr = Exists(Symbol("a"), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert repr(new_expr) == repr(expr)


def test_Symbol():
    transformer = GenericExpressionTransformer()

    expr = Symbol("apple")
    new_expr = transformer.visit(expr)
    assert new_expr is expr


def test_UnaryExpression():
    transformer = GenericExpressionTransformer()

    expr = Negation(Symbol("a"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert repr(new_expr) == repr(expr)
