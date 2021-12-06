import ast
import pytest

from pathlib import Path

from dnnv.properties import *
from dnnv.properties.dsl import PropertyParserError, parse_cli


def test_no_args_no_default():
    x = Symbol("x")
    p = Parameter("p", int)
    phi = Forall(x, x > p)

    with pytest.raises(
        PropertyParserError,
        match="No argument was provided for parameter 'p'. Try adding a command line argument '--prop.p'.",
    ):
        _ = parse_cli(phi).propagate_constants()


def test_bad_args_no_default():
    x = Symbol("x")
    p = Parameter("p", int)
    phi = Forall(x, x > p)

    with pytest.raises(
        PropertyParserError,
        match="No argument was provided for parameter 'p'. Try adding a command line argument '--prop.p'.",
    ):
        _ = parse_cli(phi, ["--bad_arg"]).propagate_constants()


def test_no_args_with_default():
    x = Symbol("x")
    p = Parameter("p", int, default=0)
    phi = Forall(x, p < x)
    phi_concrete = Forall(x, Constant(0) < x)

    phi_ = parse_cli(phi).propagate_constants()
    assert phi_.is_equivalent(phi_concrete)


def test_with_args_no_default():
    x = Symbol("x")
    p = Parameter("p", int)
    phi = Forall(x, x > p)
    phi_concrete = Forall(x, Constant(5) < x)

    phi_ = parse_cli(phi, args=["--prop.p=5"]).propagate_constants()
    assert phi_.is_equivalent(phi_concrete)
