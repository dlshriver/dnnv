from __future__ import annotations

from .symbol import Symbol


class Network(Symbol):
    def __getitem__(self, item):
        from ..base import Expression

        if not self.is_concrete:
            return super().__getitem__(item)
        if isinstance(item, Expression) and not item.is_concrete:
            return super().__getitem__(item)
        if isinstance(item, Expression):
            item = item.value
        if not isinstance(item, tuple):
            item = (item,)
        index_strs = []
        for i in item:
            if self.is_concrete and isinstance(i, int):
                index_strs.append(str(i))
            elif self.is_concrete and isinstance(i, slice):
                start = i.start if i.start is not None else ""
                stop = i.stop if i.stop is not None else ""
                s = f"{start}:{stop}"
                if i.step is not None:
                    s = f"{s}:{i.step}"
                index_strs.append(s)
        index_str = ",".join(index_strs)
        sub_network = Network(f"{self}[{index_str}]")
        if not sub_network.is_concrete:
            if len(item) == 1:
                item = item[0]
            sub_network.concretize(self.value[item])
        return sub_network

    def __repr__(self):
        return f"Network({self.identifier!r})"


__all__ = ["Network"]
