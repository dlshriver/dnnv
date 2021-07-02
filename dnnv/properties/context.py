from typing import Dict, Optional, Type, TypeVar

T = TypeVar("T")


class Context:
    _current_context = None
    count = 0

    def __init__(self):
        self.id = Context.count
        Context.count += 1
        self._instance_cache: Dict[Type[T], Dict[str, T]] = {}
        self._prev_context: Optional[Context] = None

    def __repr__(self):
        return f"Context(id={self.id}, _prev_context={self._prev_context!r})"

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__(**state)
        return self

    def __enter__(self):
        if Context._current_context is self:
            return self
        self._prev_context = Context._current_context
        Context._current_context = self
        return self

    def __exit__(self, *args):
        if self._prev_context is not None:
            Context._current_context = self._prev_context

    def reset(self):
        self._instance_cache = {}


Context._current_context = Context()


def get_context() -> Context:
    return Context._current_context
