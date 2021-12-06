from __future__ import annotations

from abc import abstractmethod
from typing import Callable, List, Optional

from . import *
from ....expressions import Expression


class FunctionSubstitutor:
    __matches__: List[Callable] = []

    @classmethod
    def lookup(cls, f: Callable) -> Optional[FunctionSubstitutor]:
        for c in cls.__subclasses__():
            if f in c.__matches__:
                return c()
        return None

    @abstractmethod
    def __call__(
        self, f: Expression, *args: Expression, **kwargs: Expression
    ) -> Expression:
        pass

    @staticmethod
    def substitute_Equal(a: Expression, b: Expression) -> Expression:
        return NotImplemented

    @staticmethod
    def substitute_NotEqual(a: Expression, b: Expression) -> Expression:
        return NotImplemented


__all__ = ["FunctionSubstitutor"]
