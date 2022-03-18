from __future__ import annotations

from typing import List

from ..expressions import Expression


class ExpressionVisitor:
    def __init__(self, recursive=False):
        self.recursive = recursive
        self.visited = {}
        self._top_level = True
        self._visit_stack = []

    def _recursive_visit(self, expression):
        method_name = "visit_%s" % expression.__class__.__name__
        visitor = getattr(self, method_name, self.generic_visit)
        self.visited[expression] = visitor(expression)
        return self.visited[expression]

    def _visit(self, expression):
        stack: List[Expression] = [expression]
        visit_stack = []
        while len(stack):
            expression = stack.pop()
            visit_stack.append(expression)
            for child_expr in expression.iter(max_depth=1, include_self=False):
                stack.append(child_expr)
        while len(visit_stack):
            expression = visit_stack.pop()
            if expression not in self.visited:
                method_name = "visit_%s" % expression.__class__.__name__
                visitor = getattr(self, method_name, self.generic_visit)
                result = visitor(expression)
                self.visited[expression] = result

    def visit(self, expression):
        is_top_level = False
        if self._top_level:
            is_top_level = True
            self._top_level = False
        if expression not in self.visited:
            if self.recursive:
                self._recursive_visit(expression)
            else:
                self._visit(expression)
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
