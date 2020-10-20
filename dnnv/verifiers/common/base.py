import logging
import os
import tempfile

from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

from dnnv.properties import Expression

from .errors import VerifierError, VerifierTranslatorError
from .executors import VerifierExecutor, CommandLineExecutor
from .reductions import (
    Property,
    Reduction,
    IOPolytopeReduction,
    HalfspacePolytope,
    HyperRectangle,
)
from .results import SAT, UNSAT, UNKNOWN, PropertyCheckResult


class Parameter:
    def __init__(
        self,
        dtype: Type,
        default: Optional[Any] = None,
        choices: Optional[List[Any]] = None,
        help: Optional[str] = None,
    ):
        self.type = dtype
        self.default = self.type(default) if default is not None else None
        self.choices = choices
        self.help = help

    def as_type(self, value):
        return self.type(value) if value is not None else None


class Verifier(ABC):
    verifier_error: Type[VerifierError] = VerifierError
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError
    executor: Type[VerifierExecutor] = CommandLineExecutor
    reduction: Union[Type[Reduction], partial] = partial(
        IOPolytopeReduction, HyperRectangle, HalfspacePolytope
    )
    parameters: Dict[str, Parameter] = {}

    def __init__(self, dnn_property: Expression, **kwargs):
        self.logger = logging.getLogger(
            f"{type(self).__module__}.{type(self).__qualname__}"
        )
        self.property = dnn_property.propagate_constants()
        for key, value in kwargs.items():
            if key not in self.__class__.parameters:
                raise self.verifier_error(f"Unknown parameter: {key}")
        self.parameters = {
            name: param.as_type(kwargs.get(name, param.default))
            for name, param in self.__class__.parameters.items()
        }

    @classmethod
    def is_installed(cls) -> bool:
        verifier = getattr(cls, "EXE", cls.__qualname__.lower())
        for path in os.environ["PATH"].split(os.pathsep):
            exe = os.path.join(path, verifier)
            if os.path.isfile(exe) and os.access(exe, os.X_OK):
                return True
        return False

    @contextmanager
    def contextmanager(self):
        yield

    @classmethod
    def verify(
        cls, phi: Expression, **kwargs
    ) -> Tuple[PropertyCheckResult, Optional[Any]]:
        return cls(phi, **kwargs).run()

    def check(self, prop: Property) -> Tuple[PropertyCheckResult, Optional[Any]]:
        with self.contextmanager():
            executor_inputs = self.build_inputs(prop)
            results = self.executor(
                *executor_inputs, verifier_error=self.verifier_error
            ).run()
        return self.parse_results(prop, results)

    def reduce_property(self) -> Generator[Property, None, None]:
        for subproperty in self.reduction(
            reduction_error=self.translator_error
        ).reduce_property(~self.property):
            yield subproperty

    def run(self) -> Tuple[PropertyCheckResult, Optional[Any]]:
        if self.property.is_concrete:
            if self.property.value == True:
                self.logger.warning("Property is trivially UNSAT.")
                return UNSAT, None
            else:
                self.logger.warning("Property is trivially SAT.")
                return SAT, None
        orig_tempdir = tempfile.tempdir
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                tempfile.tempdir = tempdir
                result = UNSAT
                for subproperty in self.reduce_property():
                    subproperty_result, cex = self.check(subproperty)
                    result |= subproperty_result
                    if result == SAT:
                        self.logger.debug("SAT! Validating counter example.")
                        if cex is not None:
                            self.validate_counter_example(subproperty, cex)
                        return result, cex
        finally:
            tempfile.tempdir = orig_tempdir
        return result, None

    def validate_counter_example(self, prop: Property, cex: Any) -> bool:
        is_valid, err_msg = prop.validate_counter_example(cex)
        if not is_valid:
            raise self.verifier_error(err_msg)
        return is_valid

    @abstractmethod
    def build_inputs(self, prop: Property) -> Tuple[Any, ...]:
        raise NotImplementedError()

    @abstractmethod
    def parse_results(
        self, prop: Property, results: Any
    ) -> Tuple[PropertyCheckResult, Optional[Any]]:
        raise NotImplementedError()


__all__ = ["Parameter", "Verifier"]
