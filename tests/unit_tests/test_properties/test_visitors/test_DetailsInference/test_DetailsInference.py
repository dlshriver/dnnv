import pytest

from dnnv.properties.expressions import *
from dnnv.properties.visitors import DetailsInference
from dnnv.properties.visitors.inference import DNNVShapeError, DNNVTypeError
from dnnv.utils import get_subclasses


def test_missing():
    class FakeExpressionType(Expression):
        pass

    with pytest.raises(
        NotImplementedError,
        match="DetailsInference is not yet implemented for expression type: FakeExpressionType",
    ) as excinfo:
        DetailsInference().visit(FakeExpressionType())


def test_non_expression():
    result = DetailsInference().visit(47)
    assert result is None
    assert Constant(47) in get_context().shapes
    assert Constant(47) in get_context().types


def test_has_all():
    visitor = DetailsInference()
    for expression_t in get_subclasses(Expression):
        if not expression_t.__module__.startswith("dnnv.properties"):
            # we don't need to support things not in DNNV
            continue
        if expression_t.__name__.endswith("Expression") or expression_t in (
            Term,
            Quantifier,
        ):
            # we don't need to have visitors for base types
            continue
        assert hasattr(visitor, f"visit_{expression_t.__name__}")


def test_set_details():
    visitor = DetailsInference()

    a = Symbol("a")
    visitor.set_details(a)
    assert not visitor.shapes[a].is_concrete
    assert not visitor.types[a].is_concrete
    visitor.set_details(a, shape=Constant(()), dtype=Constant(int))
    assert visitor.shapes[a].is_concrete
    assert visitor.types[a].is_concrete
    assert visitor.shapes[a].value == ()
    assert visitor.types[a].value == int


def test_set_details_conflicting_shapes():
    visitor = DetailsInference()

    a = Symbol("a")
    visitor.set_details(a, shape=Constant(()), dtype=Constant(int))
    assert visitor.shapes[a].is_concrete
    assert visitor.types[a].is_concrete
    assert visitor.shapes[a].value == ()
    assert visitor.types[a].value == int
    with pytest.raises(DNNVShapeError):
        visitor.set_details(a, shape=Constant((1, 2)))


def test_set_details_conflicting_types():
    visitor = DetailsInference()

    a = Symbol("a")
    visitor.set_details(a, shape=Constant(()), dtype=Constant(int))
    assert visitor.shapes[a].is_concrete
    assert visitor.types[a].is_concrete
    assert visitor.shapes[a].value == ()
    assert visitor.types[a].value == int
    with pytest.raises(DNNVTypeError):
        visitor.set_details(a, dtype=Constant(bool))


def test_set_details_symbolic():
    visitor = DetailsInference()

    a = Symbol("a")
    visitor.set_details(a)
    assert not visitor.shapes[a].is_concrete
    assert not visitor.types[a].is_concrete
    visitor.set_details(a)
    assert not visitor.shapes[a].is_concrete
    assert not visitor.types[a].is_concrete


def test_set_details_concrete():
    visitor = DetailsInference()

    a = Symbol("a")
    visitor.set_details(a, shape=Constant(()), dtype=Constant(int))
    assert visitor.shapes[a].is_concrete
    assert visitor.types[a].is_concrete
    assert visitor.shapes[a].value == ()
    assert visitor.types[a].value == int
    visitor.set_details(a, shape=Constant(()), dtype=Constant(int))
    assert visitor.shapes[a].is_concrete
    assert visitor.types[a].is_concrete
    assert visitor.shapes[a].value == ()
    assert visitor.types[a].value == int
