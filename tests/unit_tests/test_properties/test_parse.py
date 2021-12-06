from dnnv.properties import parse
from dnnv.properties.expressions import *


def test_parse_dnnp(tmp_path):
    property = """\
Forall(x, Or(x < -1, x > 1, Network("N")(x) > 1))
"""
    dnnp_path = tmp_path / "test.dnnp"

    with open(dnnp_path, "w+") as f:
        f.write(property)

    phi = parse(dnnp_path)

    print(repr(phi))
    assert (
        repr(phi)
        == "Forall(Symbol('x'), Or(GreaterThan(Network('N')(Symbol('x')), 1), GreaterThan(Symbol('x'), 1), LessThan(Symbol('x'), -1)))"
    )


def test_parse_vnnlib(tmp_path):
    property = """\
(declare-const X_0 Real)
(declare-const Y_0 Real)

(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))

(assert (<= Y_0 (- 1.0)))
"""
    vnnlib_path = tmp_path / "test.vnnlib"

    with open(vnnlib_path, "w+") as f:
        f.write(property)

    phi = parse(vnnlib_path, format="vnnlib")

    print(repr(phi))
    assert (
        repr(phi)
        == "Forall(Symbol('X'), Or(GreaterThan(Network('N')(Symbol('X'))[numpy.unravel_index(0, Network('N').'output_shape'[0])], -1.0), GreaterThan(Symbol('X')[numpy.unravel_index(0, Network('N').'input_shape'[0])], 1.0), LessThan(Symbol('X')[numpy.unravel_index(0, Network('N').'input_shape'[0])], -1.0)))"
    )
