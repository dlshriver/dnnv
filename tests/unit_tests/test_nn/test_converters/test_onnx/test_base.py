import pytest

from dnnv.nn import OperationGraph
from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *
from dnnv.utils import get_subclasses


def test_missing():
    class FakeOperation:
        pass

    with pytest.raises(ValueError) as excinfo:
        OnnxConverter(OperationGraph([])).visit(FakeOperation())
    assert str(excinfo.value).startswith(
        "ONNX converter not implemented for operation type"
    )


# def test_has_all():
#     converter = OnnxConverter(OperationGraph([]))
#     for operation in get_subclasses(Operation):
#         assert hasattr(converter, f"visit_{operation}")


# def test_convert():
#     # TODO
