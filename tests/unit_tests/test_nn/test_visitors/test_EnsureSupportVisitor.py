import pytest
import numpy as np

from dnnv.nn import OperationGraph
from dnnv.nn.operations import *
from dnnv.nn.visitors import EnsureSupportVisitor


def test_0():
    op_graph = OperationGraph([Input((1, 2, 3, 4), np.dtype(np.float64))])
    ensure_support_input = EnsureSupportVisitor(supported_operations=[Input])
    ensure_support_input.walk(op_graph)
    ensure_support_none = EnsureSupportVisitor(supported_operations=[])
    with pytest.raises(RuntimeError) as excinfo:
        ensure_support_none.walk(op_graph)
    assert str(excinfo.value) == "Input operations are not supported."


def test_1():
    op_graph = OperationGraph([Add(Input((1,), np.dtype(np.float32)), np.float32(6))])
    ensure_support_input_add = EnsureSupportVisitor(supported_operations=[Input, Add])
    ensure_support_input_add.walk(op_graph)

    ensure_support_input = EnsureSupportVisitor(supported_operations=[Input])
    with pytest.raises(RuntimeError) as excinfo:
        ensure_support_input.walk(op_graph)
    assert str(excinfo.value) == "Add operations are not supported."

    ensure_support_add = EnsureSupportVisitor(supported_operations=[Add])
    with pytest.raises(RuntimeError) as excinfo:
        ensure_support_add.walk(op_graph)
    assert str(excinfo.value) == "Input operations are not supported."

    ensure_support_none = EnsureSupportVisitor(supported_operations=[])
    with pytest.raises(RuntimeError) as excinfo:
        ensure_support_none.walk(op_graph)
    assert str(excinfo.value) == "Add operations are not supported."
