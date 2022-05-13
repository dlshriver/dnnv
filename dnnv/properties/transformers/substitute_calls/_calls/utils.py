from __future__ import annotations

import inspect
from typing import Any, Collection, Dict

from ...errors import ExpressionTransformerError


class FunctionSubstitutionError(ExpressionTransformerError):
    pass


def get_parameters(func, args, kwargs, *, supported_parameters: Collection[str]):
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    parameters: Dict[str, Any] = {}
    for name, value in bound_args.arguments.items():
        if (
            name not in supported_parameters
            and value is not sig.parameters[name].default
        ):
            raise FunctionSubstitutionError(
                f"parameter '{name}' of '{func.__qualname__}' is not currently supported"
            )
        parameters[name] = value
    return parameters
