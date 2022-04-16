import logging
import numpy as np
import os
import tempfile
import time

from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

from .errors import VerifierError, VerifierTranslatorError
from .executors import VerifierExecutor, CommandLineExecutor
from .reductions import (
    Property,
    Reduction,
    IOPolytopeReduction,
    HalfspacePolytope,
    HyperRectangle,
)
from .results import SAT, UNSAT, UNKNOWN, PropertyCheckResult, VerificationResult
from ...properties import Expression, LogicalExpression


class Parameter:
    def __init__(
        self,
        dtype: Type,
        default: Optional[Any] = None,
        choices: Optional[List[Any]] = None,
        help: Optional[str] = None,
    ):
        self.type: Callable[[Any], Any] = dtype
        if dtype == bool:
            self.type = lambda x: x not in ["False", "false", "0", "F", "f", False, 0]
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

    def __init__(
        self,
        dnn_property: Expression,
        eager_exit: bool = True,
        reraise_keyboardinterrupt=False,
        **kwargs,
    ):
        self.logger = logging.getLogger(
            f"{type(self).__module__}.{type(self).__qualname__}"
        )
        self.property = dnn_property
        self.eager_exit = eager_exit
        self.reraise_keyboardinterrupt = reraise_keyboardinterrupt
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
        if os.path.isfile(verifier) and os.access(verifier, os.X_OK):
            return True
        for path in os.environ["PATH"].split(os.pathsep):
            exe = os.path.join(path, verifier)
            if os.path.isfile(exe) and os.access(exe, os.X_OK):
                return True
        return False

    @contextmanager
    def contextmanager(self):
        yield

    @classmethod
    def verify(cls, phi: Expression, **kwargs) -> VerificationResult:
        return cls(phi, **kwargs).run()

    def check(self, prop: Property) -> VerificationResult:
        with self.contextmanager():
            executor_inputs = self.build_inputs(prop)
            start_t = time.time()
            results = self.executor(
                *executor_inputs, verifier_error=self.verifier_error
            ).run()
            end_t = time.time()
        check_result, cex = self.parse_results(prop, results)
        self.logger.debug("Check result: %s", check_result)
        if check_result == SAT:
            if cex is not None:
                self.logger.debug("SAT! Validating counter example.")
                if not self.validate_counter_example(prop, cex):
                    return VerificationResult(
                        PropertyCheckResult("unknown", "unconfirmed sat")
                    )
            else:
                self.logger.warn("SAT result without counter example.")
                return VerificationResult(
                    PropertyCheckResult("unknown", "unconfirmed sat")
                )
        return VerificationResult(check_result, cex=cex, vtime=(end_t - start_t))

    def reduce_property(self) -> Iterator[Property]:
        assert isinstance(self.property, LogicalExpression)
        for subproperty in self.reduction().reduce_property(~self.property):
            yield subproperty

    def run(self) -> VerificationResult:
        if self.property.is_concrete:
            if self.property.value == True:
                self.logger.warning("Property is trivially UNSAT.")
                return VerificationResult(UNSAT)
            else:
                self.logger.warning("Property is trivially SAT.")
                return VerificationResult(SAT)
        result = VerificationResult()
        orig_tempdir = tempfile.tempdir
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                tempfile.tempdir = tempdir
                for i, subproperty in enumerate(self.reduce_property(), start=1):
                    subproblem_result = self.check(subproperty)
                    self.logger.info("Sub-problem result %d: %s", i, subproblem_result)
                    result |= subproblem_result
                    if self.eager_exit and result.result == SAT:
                        return result
        except KeyboardInterrupt:
            if self.reraise_keyboardinterrupt:
                raise
            if result.result != SAT:
                result.result = PropertyCheckResult(
                    "unknown", "Received Signal: SIGINT"
                )
            return result
        finally:
            tempfile.tempdir = orig_tempdir
        return result

    def validate_counter_example(self, prop: Property, cex: Any) -> bool:
        is_valid, err_msg = prop.validate_counter_example(cex)
        if self.eager_exit and not is_valid:
            raise self.verifier_error(err_msg)
        return is_valid

    @abstractmethod
    def build_inputs(self, prop: Property) -> Tuple[Any, ...]:
        raise NotImplementedError()

    @abstractmethod
    def parse_results(
        self, prop: Property, results: Any
    ) -> Tuple[PropertyCheckResult, Optional[np.ndarray]]:
        raise NotImplementedError()


__all__ = ["Parameter", "Verifier"]
