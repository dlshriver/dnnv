from dnnv.nn.operations.patterns import *


def test_and():
    or_and_or = Or() & Or()
    assert isinstance(or_and_or, Parallel)

    or_and_none = Or() & None
    assert isinstance(or_and_none, Parallel)

    seq_and_seq = Sequential() & Sequential()
    assert isinstance(seq_and_seq, Parallel)

    seq_and_none = Sequential() & None
    assert isinstance(seq_and_none, Parallel)


def test_rand():
    none_and_or = None & Or()
    assert isinstance(none_and_or, Parallel)

    none_and_seq = None & Sequential()
    assert isinstance(none_and_seq, Parallel)


def test_or():
    par_or_par = Parallel() | Parallel()
    assert isinstance(par_or_par, Or)

    par_or_none = Parallel() | None
    assert isinstance(par_or_none, Or)

    seq_or_seq = Sequential() | Sequential()
    assert isinstance(seq_or_seq, Or)

    seq_or_none = Sequential() | None
    assert isinstance(seq_or_none, Or)


def test_ror():
    none_or_par = None | Parallel()
    assert isinstance(none_or_par, Or)

    none_or_seq = None | Sequential()
    assert isinstance(none_or_seq, Or)


def test_rshift():
    or_rhift_or = Or() >> Or()
    assert isinstance(or_rhift_or, Sequential)

    or_rhift_none = Or() >> None
    assert isinstance(or_rhift_none, Sequential)

    par_rhift_par = Parallel() >> Parallel()
    assert isinstance(par_rhift_par, Sequential)

    par_rhift_none = Parallel() >> None
    assert isinstance(par_rhift_none, Sequential)


def test_rrshift():
    none_rhift_or = None >> Or()
    assert isinstance(none_rhift_or, Sequential)

    none_rhift_par = None >> Parallel()
    assert isinstance(none_rhift_par, Sequential)
