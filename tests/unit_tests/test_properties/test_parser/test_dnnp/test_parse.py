from pathlib import Path

import pytest

from dnnv.properties.expressions import Expression, Forall, get_context
from dnnv.properties.parser.dnnp import DNNPParserError, parse, parse_str

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
        == "Forall(Symbol('x'), Implies(And(GreaterThan(Network('N')(Symbol('x')), 1), LessThanOrEqual(0, Symbol('x')), LessThanOrEqual(Symbol('x'), 1)), LessThan(Network('N')(Symbol('x')), Multiply(2, Network('N')(Symbol('x'))))))"
    )


def test_unsupported_structures():
    spec_str = """from dnnv.properties import *
c = 0
if True:
    c = 5
Forall(x, x > c)
"""
    with pytest.raises(
        DNNPParserError, match="line 3, column 0: Unsupported structure in property:"
    ):
        _ = parse_str(spec_str)


def test_no_expr():
    spec_str = """from dnnv.properties import *
c = 0
"""
    with pytest.raises(DNNPParserError, match="No property expression found"):
        _ = parse_str(spec_str)
