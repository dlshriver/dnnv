from __future__ import annotations

from .constant import Constant
from .symbol import Symbol


class Network(Symbol):
    def __getitem__(self, item):
        if isinstance(item, Constant):
            item = item.value
        if self.is_concrete and isinstance(item, int):
            sub_network = Network(f"{self}[{item}]")
            if not sub_network.is_concrete:
                sub_network.concretize(self.value[item])
            return sub_network
        if self.is_concrete and isinstance(item, slice):
            start = item.start if item.start is not None else ""
            stop = item.stop if item.stop is not None else ""
            s = f"{start}:{stop}"
            if item.step is not None:
                s = f"{s}:{item.step}"
            sub_network = Network(f"{self}[{s}]")
            if not sub_network.is_concrete:
                sub_network.concretize(self.value[item])
            return sub_network
        return super().__getitem__(item)

    def __repr__(self):
        return f"Network({self.identifier!r})"


__all__ = ["Network"]
