from typing import Dict, List, Type, TypeVar

T = TypeVar("T")


class Context:
    _current_context = None
    count = 0

    def __init__(self):
        self.id = Context.count
        Context.count += 1
        self._instance_cache: Dict[Type[T], Dict[str, T]] = {}
        self._prev_contexts: List[Context] = []

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


Context._current_context = Context()


def get_context() -> Context:
    return Context._current_context


__all__ = ["get_context", "Context"]
