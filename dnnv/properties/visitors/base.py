from __future__ import annotations

from ..expressions import Expression


class ExpressionVisitor:
    def __init__(self):
        self.visited = {}
        self._top_level = True

    def visit(self, expression):
        is_top_level = False
        if self._top_level:
            is_top_level = True
            self._top_level = False
        if expression not in self.visited:
            method_name = "visit_%s" % expression.__class__.__name__
            visitor = getattr(self, method_name, self.generic_visit)
            self.visited[expression] = visitor(expression)
        if is_top_level:
            self._top_level = True
        return self.visited[expression]

    def generic_visit(self, expression):
        for value in expression.__dict__.values():
            if isinstance(value, Expression):
                self.visit(value)
            elif isinstance(value, (list, tuple, set)):
                for sub_value in value:
                    if isinstance(sub_value, Expression):
                        self.visit(sub_value)
        return expression


__all__ = ["ExpressionVisitor"]
