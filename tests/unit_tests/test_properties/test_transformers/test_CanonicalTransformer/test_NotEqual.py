from dnnv.properties import *
from dnnv.properties.transformers import CanonicalTransformer


def test_NotEqual_symbols():
    transformer = CanonicalTransformer()

    a, b = Symbol("a"), Symbol("b")
    expr = NotEqual(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        Or(
            And(
                LessThan(
                    Add(
                        Multiply(Constant(-1), Symbol("a")),
                        Multiply(Constant(1), Symbol("b")),
                    ),
                    Constant(0),
                )
            ),
            And(
                LessThan(
                    Add(
                        Multiply(Constant(-1), Symbol("b")),
                        Multiply(Constant(1), Symbol("a")),
                    ),
                    Constant(0),
                )
            ),
        )
    )


def test_NotEqual_constants():
    transformer = CanonicalTransformer()
    transformer._top_level = False

    expr = NotEqual(Constant(302), Constant(120))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 2
    assert isinstance(new_expr.expressions[0], And)
    assert len(new_expr.expressions[0].expressions) == 1
    or_expressions = new_expr.expressions
    assert And(GreaterThan(Add(), Constant(-182))) in or_expressions
    assert And(GreaterThan(Add(), Constant(182))) in or_expressions
