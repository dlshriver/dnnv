from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Tuple, Type, Union

from dnnv.errors import DNNVError
from dnnv.properties import Expression

from ..results import PropertyCheckResult


class ReductionError(DNNVError):
    pass


class Property(ABC):
    @abstractmethod
    def is_trivial(
        self,
    ) -> Union[Tuple[bool], Tuple[bool, Tuple[PropertyCheckResult, Any]]]:
        raise NotImplementedError()

    @abstractmethod
    def validate_counter_example(self, cex: Any) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError()


class Reduction(ABC):
    def __init__(
        self,
        reduction_error: Type[ReductionError] = ReductionError,
    ):
        self.reduction_error = reduction_error

    @abstractmethod
    def reduce_property(self, expression: Expression) -> Iterator[Property]:
        raise NotImplementedError()


__all__ = ["Property", "Reduction", "ReductionError"]
