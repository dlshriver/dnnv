from abc import ABC, abstractmethod
from typing import Any, Generator, Optional, Tuple, Type

from dnnv.properties import Expression

from ..errors import VerifierTranslatorError


class Property:
    @abstractmethod
    def validate_counter_example(self, cex: Any) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError()


class Reduction(ABC):
    def __init__(
        self,
        reduction_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
    ):
        self.reduction_error = reduction_error

    @abstractmethod
    def reduce_property(self, phi: Expression) -> Generator[Property, None, None]:
        raise NotImplementedError()


__all__ = ["Property", "Reduction"]
