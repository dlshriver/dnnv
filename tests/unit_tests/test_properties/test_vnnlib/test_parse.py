from dnnv.properties.base import *
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

    assert isinstance(phi, Forall)


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

    assert isinstance(phi, Forall)


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

    assert isinstance(phi, Forall)
