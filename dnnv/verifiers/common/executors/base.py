from abc import ABC, abstractmethod
from typing import Any

from ..errors import VerifierError


class VerifierExecutor(ABC):
    def __init__(self, *args, verifier_error=VerifierError):
        self.args = args
        self.verifier_error = verifier_error

    @abstractmethod
    def run(self) -> Any:
        raise NotImplementedError()


__all__ = ["VerifierExecutor"]
