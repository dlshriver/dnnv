from __future__ import annotations

from .constant import Constant
from .symbol import Symbol
from ..call import Compose


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

    def compose(self, other):
        return Compose(self, other)

    @classmethod
    def from_op_graph(cls, op_graph):
        name = getattr(op_graph, "_dnnv_network_id", f"N_{id(op_graph):x}")
        new = cls(name)
        new.concretize(op_graph)
        return new


__all__ = ["Network"]
