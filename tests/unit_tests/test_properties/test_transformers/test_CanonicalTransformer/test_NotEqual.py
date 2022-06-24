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
    assert new_expr is Constant(True)
