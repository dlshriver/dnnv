from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..context import Context
from .base import Term
from .utils import get_callable_name


class Constant(Term):
    def __init__(self, value: Any, ctx: Optional[Context] = None):
        super().__init__(ctx=ctx)
        self._value = value
        self._identifier = Constant.build_identifier(value)
        if isinstance(value, np.ndarray):
            value.setflags(write=False)

    @staticmethod
    def build_identifier(value, **kwargs):
        if isinstance(value, Constant):
            value = value._value
        if isinstance(
            value,
            (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            value = int(value)
        value_type = type(value)
        try:
            _ = hash(value)
            value_identifier = value
        except:
            if isinstance(value, np.ndarray):
                value_identifier = (value.tobytes(), value.shape, value.dtype)
            else:
                value_identifier = id(value)
        return value_type, value_identifier

    @property
    def is_concrete(self) -> bool:
        return True

    @property
    def value(self):
        return self._value

    def __bool__(self):
        return bool(self._value)

    def __hash__(self):
        return super().__hash__() * hash(self._identifier)

    def __repr__(self):
        value = self.value
        if isinstance(value, (str, int, float, dict, list, set, tuple)):
            return repr(value)
        if isinstance(value, slice):
            start = value.start if value.start is not None else ""
            stop = value.stop if value.stop is not None else ""
            if value.step is not None:
                return f"{start}:{stop}:{value.step}"
            return f"{start}:{stop}"
        if callable(value):
            return get_callable_name(value)
        if isinstance(value, np.ndarray):
            _, (_, shape, dtype) = self._identifier
            return f"np.ndarray{{id={hex(id(value))}, shape={shape}, dtype={dtype}}}"
        return repr(self._identifier)

    def __str__(self):
        value = self.value
        if isinstance(value, (str, int, float)):
            return repr(value)
        if isinstance(value, np.ndarray):
            return "".join(
                np.array2string(
                    value,
                    max_line_width=np.inf,
                    precision=3,
                    threshold=5,
                    edgeitems=2,
                ).split("\n")
            ).replace("  ", " ")
        if isinstance(value, slice):
            start = value.start if value.start is not None else ""
            stop = value.stop if value.stop is not None else ""
            if value.step is not None:
                return f"{start}:{stop}:{value.step}"
            return f"{start}:{stop}"
        if callable(value):
            return get_callable_name(value)
        return str(value)


__all__ = ["Constant"]
