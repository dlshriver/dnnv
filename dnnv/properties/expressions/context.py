from __future__ import annotations

from typing import Any, Dict, Hashable, List, Optional, Sequence, Type, TypeVar

from .. import expressions

T = TypeVar("T")


class Context:
    _current_context: Optional[Context] = None
    count = 0

    def __init__(self):
        self.id = Context.count
        Context.count += 1
        self._instance_cache: Dict[Type, Dict[Hashable, Any]] = {}
        self._prev_contexts: List[Context] = []

        self.shapes: Dict[expressions.Expression, Sequence[int]] = {}
        self.types: Dict[expressions.Expression, Any] = {}

    def __repr__(self):
        prev_contexts = ", ".join(str(ctx) for ctx in self._prev_contexts)
        return f"Context(id={self.id}, prev_contexts=[{prev_contexts}])"

    def __str__(self):
        return f"Context(id={self.id})"

    def __enter__(self):
        self._prev_contexts.append(Context._current_context)
        Context._current_context = self
        return self

    def __exit__(self, *args):
        Context._current_context = self._prev_contexts.pop()

    def reset(self):
        self._instance_cache = {}
        self.shapes = {}
        self.types = {}


Context._current_context = Context()


def get_context() -> Context:
    assert Context._current_context is not None
    return Context._current_context


__all__ = ["get_context", "Context"]
