import pytest
import numpy as np

from dnnv.nn import OperationGraph
from dnnv.nn.operations import *
from dnnv.nn.visitors import EnsureSupportVisitor


def test_0():
    op_graph = OperationGraph([Input((1, 2, 3, 4), np.dtype(np.float64))])
    ensure_support_input = EnsureSupportVisitor(supported_operations=[Input])
    op_graph.walk(ensure_support_input)
    ensure_support_none = EnsureSupportVisitor(supported_operations=[])
    with pytest.raises(RuntimeError) as excinfo:
        op_graph.walk(ensure_support_none)
    assert str(excinfo.value) == "Input operations are not supported."


def test_1():
    op_graph = OperationGraph([Add(Input((1,), np.dtype(np.float32)), np.float32(6))])
    ensure_support_input_add = EnsureSupportVisitor(supported_operations=[Input, Add])
    op_graph.walk(ensure_support_input_add)

    ensure_support_input = EnsureSupportVisitor(supported_operations=[Input])
    with pytest.raises(RuntimeError) as excinfo:
        op_graph.walk(ensure_support_input)
    assert str(excinfo.value) == "Add operations are not supported."

    ensure_support_add = EnsureSupportVisitor(supported_operations=[Add])
    with pytest.raises(RuntimeError) as excinfo:
        op_graph.walk(ensure_support_add)
    assert str(excinfo.value) == "Input operations are not supported."

    ensure_support_none = EnsureSupportVisitor(supported_operations=[])
    with pytest.raises(RuntimeError) as excinfo:
        op_graph.walk(ensure_support_none)
    assert str(excinfo.value) == "Add operations are not supported."
