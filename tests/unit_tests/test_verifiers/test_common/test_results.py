import pytest

from dnnv.verifiers.common.results import *
from dnnv.verifiers.common.results import PropertyCheckResult


def test_str():
    assert str(SAT) == "sat"
    assert str(UNSAT) == "unsat"
    assert str(UNKNOWN) == "unknown"
    assert str(PropertyCheckResult("NEW_RESULT_TYPE")) == "NEW_RESULT_TYPE"


def test_repr():
    assert repr(SAT) == "PropertyCheckResult('sat')"
    assert repr(UNSAT) == "PropertyCheckResult('unsat')"
    assert repr(UNKNOWN) == "PropertyCheckResult('unknown')"
    assert (
        repr(PropertyCheckResult("NEW_RESULT_TYPE"))
        == "PropertyCheckResult('NEW_RESULT_TYPE')"
    )


def test_invert():
    assert ~SAT == UNSAT
    assert ~UNSAT == SAT
    assert ~UNKNOWN == UNKNOWN
    assert ~PropertyCheckResult("NEW_RESULT_TYPE") == UNKNOWN


def test_and():
    assert (SAT & SAT) == SAT
    assert (SAT & UNSAT) == UNSAT
    assert (SAT & UNKNOWN) == UNKNOWN
    assert (PropertyCheckResult("NEW_RESULT_TYPE") & SAT) == UNKNOWN

    with pytest.raises(TypeError):
        _ = SAT & 1


def test_or():
    assert (UNSAT | SAT) == SAT
    assert (UNSAT | UNSAT) == UNSAT
    assert (UNSAT | UNKNOWN) == UNKNOWN
    assert (PropertyCheckResult("NEW_RESULT_TYPE") | SAT) == UNKNOWN

    with pytest.raises(TypeError):
        _ = SAT | 1


def test_eq():
    assert SAT == SAT
    assert not (SAT == UNSAT)
    assert not (SAT == UNKNOWN)
    assert not (SAT == PropertyCheckResult("NEW_RESULT_TYPE"))
    assert UNSAT == UNSAT
    assert not (UNSAT == UNKNOWN)
    assert not (UNSAT == PropertyCheckResult("NEW_RESULT_TYPE"))
    assert UNKNOWN == UNKNOWN
    assert not (UNKNOWN == PropertyCheckResult("NEW_RESULT_TYPE"))
    assert PropertyCheckResult("NEW_RESULT_TYPE") == PropertyCheckResult(
        "NEW_RESULT_TYPE"
    )
    assert SAT != 1
