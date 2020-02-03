from ..operations import Operation
from ..visitors import OperationVisitor


class OperationTransformer(OperationVisitor):
    def generic_visit(self, operation: Operation) -> Operation:
        for name, value in operation.__dict__.items():
            if isinstance(value, Operation):
                new_value = self.visit(value)
                setattr(operation, name, new_value)
            elif isinstance(value, (list, set, tuple)):
                new_value = []
                for value_ in value:
                    if isinstance(value_, Operation):
                        new_value_ = self.visit(value_)
                        new_value.append(new_value_)
                    else:
                        new_value.append(value_)
                setattr(operation, name, type(value)(new_value))
        return operation


__all__ = ["OperationTransformer"]
