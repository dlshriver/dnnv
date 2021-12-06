import pytest

from dnnv.properties.expressions import *
from dnnv.properties.vnnlib import ExpressionBuilder, VNNLibParseError


def test_bad_parse():
    with pytest.raises(VNNLibParseError, match="Parsing failed at index 0"):
        _ = ExpressionBuilder().build("(+ 3 2 1)")


def test_multi_arg_subtract():
    with pytest.raises(
        VNNLibParseError, match="Subtraction not implemented for more than 2 args."
    ):
        _ = ExpressionBuilder().build("(assert (= (- 3 2 1) 0))")


def test_declare_twice():
    with pytest.raises(
        VNNLibParseError, match="Name already exists in symbol table: varname"
    ):
        _ = ExpressionBuilder().build(
            "(declare-const varname Real)\n(declare-const varname Real)"
        )


def test_unknown_identifier():
    with pytest.raises(VNNLibParseError, match="Unknown identifier: badidentifier"):
        _ = ExpressionBuilder().build("(assert (badidentifier 4))")


def test_unknown_sort():
    with pytest.raises(NotImplementedError, match="Unimplemented sort: badsort"):
        _ = ExpressionBuilder().build("(declare-const X badsort)")
