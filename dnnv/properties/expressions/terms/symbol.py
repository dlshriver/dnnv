from __future__ import annotations

from typing import Any, Optional, Union

from ..context import Context
from .base import Term
from .constant import Constant


class Symbol(Term):
    _ssa_indices = {}

    def __init__(self, identifier: Union[Constant, str], ctx: Optional[Context] = None):
        super().__init__(ctx=ctx)
        if isinstance(identifier, Constant):
            self.identifier = identifier.value
        else:
            self.identifier = identifier
        self._value = None

    def build_identifier(identifier, *args, **kwargs):
        if isinstance(identifier, Constant):
            identifier = identifier.value
        if not isinstance(identifier, str):
            raise TypeError("Argument 'identifier' for Symbol must be of type str")
        return identifier

    @property
    def is_concrete(self):
        return self._value is not None

    @property
    def value(self):
        if not self.is_concrete:
            raise ValueError("Cannot get value of non-concrete symbol.")
        return self._value

    def __hash__(self):
        return super().__hash__() * hash(self.identifier)

    def __repr__(self):
        return f"Symbol({self.identifier!r})"

    def __str__(self):
        return self.identifier


__all__ = ["Symbol"]
