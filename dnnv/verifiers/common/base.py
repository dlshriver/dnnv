import logging
import os
import tempfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from dnnv.properties import Expression, LogicalExpression

from .errors import VerifierError, VerifierTranslatorError
from .executors import CommandLineExecutor, VerifierExecutor
from .reductions import (
    HalfspacePolytope,
    HyperRectangle,
    IOPolytopeReduction,
    Property,
    Reduction,
)
from .results import SAT, UNSAT, PropertyCheckResult


class Parameter:
    def __init__(
        self,
        dtype: Callable[[Any], Any],
        default: Optional[Any] = None,
        choices: Optional[List[Any]] = None,
        help: Optional[str] = None,
    ):
        self.type: Callable[[Any], Any] = dtype
        if dtype == bool:
            self.type = lambda x: x not in [
                "False",
                "false",
                "0",
                "F",
                "f",
                False,
                0,
            ]
        self.default = self.type(default) if default is not None else None
        self.choices = choices
        self.help = help

    def as_type(self, value: Any):
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
        for key in kwargs:
            if key not in self.__class__.parameters:
                raise self.verifier_error(f"Unknown parameter: {key}")
        self.parameter_values: Dict[str, Any] = {
            name: param.as_type(kwargs.get(name, param.default))
            for name, param in self.__class__.parameters.items()
        }

    @classmethod
    def is_installed(cls) -> bool:
        verifier = getattr(cls, "EXE", cls.__qualname__.lower())
        if os.path.isfile(verifier) and os.access(verifier, os.X_OK):
            return True
        for path in os.environ["PATH"].split(os.pathsep):
            exe = os.path.join(path, verifier)
            if os.path.isfile(exe) and os.access(exe, os.X_OK):
                return True
        return False

    @contextmanager
    def contextmanager(self):
        yield self

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
        assert isinstance(self.property, LogicalExpression)
        for subproperty in self.reduction().reduce_property(~self.property):
            yield subproperty

    def run(self) -> Tuple[PropertyCheckResult, Optional[Any]]:
        if self.property.is_concrete:
            if self.property.value:
                self.logger.warning("Property is trivially UNSAT.")
                return UNSAT, None
            self.logger.warning("Property is trivially SAT.")
            return SAT, None
        orig_tempdir = tempfile.tempdir
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                tempfile.tempdir = tempdir
                result = UNSAT
                for subproperty in self.reduce_property():
                    is_trivial, *trivial_result = subproperty.is_trivial()
                    if is_trivial:
                        subproperty_result, cex = trivial_result[0]
                        self.logger.warning(
                            "Property is trivially %s.", subproperty_result
                        )
                    else:
                        subproperty_result, cex = self.check(subproperty)
                    result |= subproperty_result
                    if result == SAT:
                        if cex is not None:
                            self.logger.debug("SAT! Validating counter example.")
                            self.validate_counter_example(subproperty, cex)
                        else:
                            self.logger.warning("SAT result without counter example.")
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
    def build_inputs(self, prop: Property) -> Sequence:  # pragma: no cover
        raise NotImplementedError()

    @abstractmethod
    def parse_results(
        self, prop: Property, results: Any
    ) -> Tuple[PropertyCheckResult, Optional[Any]]:  # pragma: no cover
        raise NotImplementedError()


__all__ = ["Parameter", "Verifier"]
