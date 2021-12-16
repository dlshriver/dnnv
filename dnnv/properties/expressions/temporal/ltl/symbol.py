from __future__ import annotations

from typing import Optional, Union

from .base import LTL, LTLUnwindingBoundReached
from ...call import Compose
from ...terms import Constant, Network, Parameter, Symbol
from .....nn import operations, OperationGraph, OperationVisitor
from .....nn.transformers.mlp import MLPTransformer


class TemporalSymbol(LTL):
    def __init__(
        self,
        symbol: Symbol,
        network: Network,
        transition_fn: Network,
        state_map_fn: Optional[Network] = None,
        *,
        step: Optional[Union[int, Constant]] = None,
        unwind: Optional[Union[int, Parameter]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.symbol = symbol
        self.network = network
        self.transition_fn = transition_fn
        self.state_map_fn = state_map_fn

        self.step = step
        if step is not None and isinstance(step, Constant):
            self.step = step.value

        self.unwind = unwind
        if unwind is not None and isinstance(unwind, Parameter) and unwind.is_concrete:
            self.unwind = unwind.value

        self._stepper: Optional[Network] = None

    def __repr__(self):
        state_map_fn = ""
        if self.state_map_fn is not None:
            state_map_fn = f", {self.state_map_fn!r}"
        step = ""
        if self.step is not None:
            step = f", {self.step!r}"
        if self.unwind is not None:
            unwind = f", {self.unwind!r}"
        return f"TemporalSymbol({self.symbol!r}, {self.network!r}, {self.transition_fn!r}{state_map_fn}{step}{unwind})"

    def __str__(self):
        step = self.step if self.step is not None else "t"
        return f"{self.symbol}_{step}"

    def stepper(self):
        if self._stepper is None:
            if self.state_map_fn is not None:
                network = Network(
                    f"!({self.network.identifier}○{self.state_map_fn.identifier})"
                )
                assert self.state_map_fn.is_concrete
                assert self.network.is_concrete

                get_op_graph = lambda: self.network.value.compose(
                    self.state_map_fn.value
                )
            else:
                network = Network(f"!({self.network.identifier})")
                get_op_graph = lambda: self.network.value

            if not network.is_concrete:
                op_graph = get_op_graph()

                class GetInputs(OperationVisitor):
                    def __init__(self):
                        super().__init__()
                        self.inputs = set()

                    def walk(self, op_graph: OperationGraph):
                        op_graph.walk(self)
                        return self

                    def visit_Input(self, op):
                        self.inputs.add(op)

                inputs = GetInputs().walk(op_graph).inputs
                assert len(inputs) == 1

                new_output_op = operations.Concat(
                    (inputs.pop(),) + op_graph.output_operations,
                    1,
                )
                new_op_graph = OperationGraph([new_output_op])

                # TODO : can we save this for later?
                new_mlp_op_graph = MLPTransformer().transform(new_op_graph)
                network.concretize(new_mlp_op_graph)

            assert network.is_concrete

            stepper = Compose(self.transition_fn, network)
            if self.state_map_fn is not None:
                stepper = Compose(stepper, self.state_map_fn)
            self._stepper = stepper.propagate_constants()
        return self._stepper

    def freeze(self):
        if not self.step:
            return self.symbol

        stepper = self.stepper()
        transition = stepper
        for _ in range(1, self.step or 0):
            transition = Compose(transition, stepper)
        transition = transition.propagate_constants()
        return transition(self.symbol)

    def next(self):
        step = self.step or 0
        if self.unwind is None:
            raise ValueError("No bound specified for unwinding")
        if step < self.unwind:
            next_symbol = TemporalSymbol(
                self.symbol,
                self.network,
                self.transition_fn,
                self.state_map_fn,
                step=step + 1,
                unwind=self.unwind,
                ctx=self.ctx,
            )
            next_symbol._stepper = self.stepper()
            return next_symbol
        raise LTLUnwindingBoundReached()


__all__ = ["TemporalSymbol"]
