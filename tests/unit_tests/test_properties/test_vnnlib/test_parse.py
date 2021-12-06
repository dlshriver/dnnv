from dnnv.properties.expressions import *
from dnnv.properties.vnnlib import parse


def test_parse_0(tmp_path):
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

    phi = parse(vnnlib_path)

    assert (
        repr(phi)
        == "Forall(Symbol('X'), Or(GreaterThan(Network('N')(Symbol('X'))[numpy.unravel_index(0, Network('N').'output_shape'[0])], -1.0), GreaterThan(Symbol('X')[numpy.unravel_index(0, Network('N').'input_shape'[0])], 1.0), LessThan(Symbol('X')[numpy.unravel_index(0, Network('N').'input_shape'[0])], -1.0)))"
    )


def test_parse_1(tmp_path):
    property = """\
(declare-const X_0 Int)
(declare-const Y_0 Int)

(assert (>= X_0 -1))
(assert (<= X_0 1))

(assert (<= (+ 2 Y_0) -100))
(assert (<= (- 2 Y_0) 0))
"""
    vnnlib_path = tmp_path / "test.vnnlib"

    with open(vnnlib_path, "w+") as f:
        f.write(property)

    phi = parse(vnnlib_path)

    assert (
        repr(phi)
        == "Forall(Symbol('X'), Or(GreaterThan(Add(2, Network('N')(Symbol('X'))[numpy.unravel_index(0, Network('N').'output_shape'[0])]), -100), GreaterThan(Subtract(2, Network('N')(Symbol('X'))[numpy.unravel_index(0, Network('N').'output_shape'[0])]), 0), GreaterThan(Symbol('X')[numpy.unravel_index(0, Network('N').'input_shape'[0])], 1), LessThan(Symbol('X')[numpy.unravel_index(0, Network('N').'input_shape'[0])], -1)))"
    )


def test_parse_2(tmp_path):
    property = """\
(declare-const X_0 Bool)
(declare-const Y_0 Bool)

(assert (>= X_0 0))
(assert (<= X_0 1))

(assert (<= Y_0 0))
"""
    vnnlib_path = tmp_path / "test.vnnlib"

    with open(vnnlib_path, "w+") as f:
        f.write(property)

    phi = parse(vnnlib_path)

    assert (
        repr(phi)
        == "Forall(Symbol('X'), Or(GreaterThan(Network('N')(Symbol('X'))[numpy.unravel_index(0, Network('N').'output_shape'[0])], 0), GreaterThan(Symbol('X')[numpy.unravel_index(0, Network('N').'input_shape'[0])], 1), LessThan(Symbol('X')[numpy.unravel_index(0, Network('N').'input_shape'[0])], 0)))"
    )


def test_parse_multidim(tmp_path):
    property = """\
(declare-const X_0_0 Real)
(declare-const X_0_1 Real)
(declare-const Y_0_0 Real)
(declare-const Y_0_1 Real)

(assert (>= X_0_0 -1.0))
(assert (<= X_0_0 1.0))
(assert (>= X_0_1 -1.0))
(assert (<= X_0_1 1.0))

(assert (<= Y_0_0 (- 1.0)))
(assert (<= Y_0_1 (- 1.0)))
"""
    vnnlib_path = tmp_path / "test.vnnlib"

    with open(vnnlib_path, "w+") as f:
        f.write(property)

    phi = parse(vnnlib_path)

    assert (
        repr(phi)
        == "Forall(Symbol('X'), Or(GreaterThan(Network('N')(Symbol('X'))[(0, 0)], -1.0), GreaterThan(Network('N')(Symbol('X'))[(0, 1)], -1.0), GreaterThan(Symbol('X')[(0, 0)], 1.0), GreaterThan(Symbol('X')[(0, 1)], 1.0), LessThan(Symbol('X')[(0, 0)], -1.0), LessThan(Symbol('X')[(0, 1)], -1.0)))"
    )


def test_parse_parameters(tmp_path):
    property = """\
(declare-const X_0_0 Real)
(declare-const X_0_1 Real)

(declare-const Y_0_0 Real)

(declare-const epsilon Real)

(assert (>= X_0_0 (- epsilon)))
(assert (<= X_0_0 epsilon))
(assert (>= X_0_1 (- epsilon)))
(assert (<= X_0_1 epsilon))

(assert (> Y_0_0 0.0))
"""
    vnnlib_path = tmp_path / "test.vnnlib"

    with open(vnnlib_path, "w+") as f:
        f.write(property)

    phi = parse(vnnlib_path, ["--prop.epsilon=1.0"]).propagate_constants()

    assert (
        repr(phi)
        == "Forall(Symbol('X'), Or(GreaterThan(Symbol('X')[(0, 0)], 1.0), GreaterThan(Symbol('X')[(0, 1)], 1.0), LessThan(Symbol('X')[(0, 0)], -1.0), LessThan(Symbol('X')[(0, 1)], -1.0), LessThanOrEqual(Network('N')(Symbol('X'))[(0, 0)], 0.0)))"
    )
