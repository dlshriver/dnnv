from __future__ import annotations

from ..errors import DNNVError


class DNNVExpressionError(DNNVError):
    pass


class NonConcreteExpressionError(DNNVExpressionError):
    pass


__all__ = ["DNNVExpressionError", "NonConcreteExpressionError"]
