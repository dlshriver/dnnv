from dnnv.nn.operations import *
from dnnv.nn.operations.base import *


def test_str():
    assert str(Operation) == "Operation"
    assert str(Input) == "Input"
    assert str(Add) == "Add"
    assert str(Gemm) == "Gemm"


def test_and():
    op_and_op = Operation & Operation
    assert isinstance(op_and_op, Parallel)

    op_and_none = Operation & None
    assert isinstance(op_and_none, Parallel)


def test_rand():
    none_and_op = None & Operation
    assert isinstance(none_and_op, Parallel)


def test_or():
    op_or_op = Operation | Operation
    assert isinstance(op_or_op, Or)

    op_or_none = Operation | None
    assert isinstance(op_or_none, Or)


def test_ror():
    none_or_op = None | Operation
    assert isinstance(none_or_op, Or)


def test_rshift():
    op_rhift_op = Operation >> Operation
    assert isinstance(op_rhift_op, Sequential)

    op_rhift_none = Operation >> None
    assert isinstance(op_rhift_none, Sequential)


def test_rrshift():
    none_rshift_op = None >> Operation
    assert isinstance(none_rshift_op, Sequential)
