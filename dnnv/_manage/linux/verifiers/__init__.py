from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

verifier_choices = [verifier.name for verifier in pkgutil.iter_modules(__path__)]


def import_verifier_module(name: str) -> ModuleType:
    return importlib.import_module(f"{__package__}.{name}")


__all__ = ["import_verifier_module", "verifier_choices"]
