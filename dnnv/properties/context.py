from typing import Dict, Optional, Type, TypeVar

T = TypeVar("T")


class Context:
    count = 0

    def __init__(self):
        self.id = Context.count
        Context.count += 1
        self._instance_cache: Dict[Type[T], Dict[str, T]] = {}
        self._prev_context: Optional[Context] = None

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__(**state)
        return self

    def __enter__(self):
        global _current_context
        self._prev_context = _current_context
        _current_context = self
        return self

    def __exit__(self, *args):
        global _current_context
        _current_context = self._prev_context

    def reset(self):
        self._instance_cache = {}


_current_context = Context()


def get_context() -> Context:
    return _current_context
