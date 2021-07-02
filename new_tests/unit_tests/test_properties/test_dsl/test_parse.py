from dnnv.properties.base import Expression, Forall
from pathlib import Path

from dnnv.properties.context import get_context
from dnnv.properties.dsl import parse

artifacts_dir = Path(__file__).parent / "test_parse_artifacts"


def test_true():
    phi = parse(artifacts_dir / "true.dnnp")
    assert isinstance(phi, Expression)
    assert phi.ctx != get_context()
    assert isinstance(phi, Forall)
    assert len(phi.networks) == 1
    assert len(phi.variables) == 2
    assert (
        repr(phi)
        == "Forall(Symbol('x0'), Implies(And(GreaterThan(Network('N')(Symbol('x0')), 1), LessThanOrEqual(0, Symbol('x0')), LessThanOrEqual(Symbol('x0'), 1)), LessThan(Network('N')(Symbol('x0')), Multiply(2, Network('N')(Symbol('x0'))))))"
    )
