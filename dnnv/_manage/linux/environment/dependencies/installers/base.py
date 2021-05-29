from __future__ import annotations

import typing

from ..base import Dependency

if typing.TYPE_CHECKING:
    from ...base import Environment


class Installer:
    def run(self, env: Environment, dependency: Dependency):
        raise NotImplementedError()


__all__ = ["Installer"]
