from __future__ import annotations

import numpy as np

from typing import Any, Dict, Optional, Tuple, Union

from .base import LTL, LTLUnwindingBoundReached
from ...call import Compose
from ...terms import Constant, Network, Parameter, Symbol
from .....nn import operations, OperationGraph, OperationVisitor
from .....nn.transformers.mlp import MLPTransformer
from .....nn.utils import TensorDetails


class TemporalSymbol(LTL):
    def __init__(
        self,
        symbol: Symbol,
        network: Network,
        transition_fn: Network,
        state_map_fn: Optional[Network] = None,
        *,
        unwind: Optional[Union[int, Constant, Parameter]] = None,
        num_noise_vars: Optional[Union[int, Constant, Parameter]] = 0,
        current_step: Optional[Union[int, Constant]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.symbol = symbol
        self.network = network
        self.transition_fn = transition_fn
        self.state_map_fn = state_map_fn

        if current_step is not None and isinstance(current_step, Constant):
            current_step = current_step.value
        self.step = current_step

        if (
            unwind is not None
            and isinstance(unwind, (Constant, Parameter))
            and unwind.is_concrete
        ):
            unwind = unwind.value
        self.unwind = unwind

        if (
            isinstance(num_noise_vars, (Constant, Parameter))
            and num_noise_vars.is_concrete
        ):
            num_noise_vars = num_noise_vars.value
        self.num_noise_vars = num_noise_vars

        self._cache: Dict[Any, Any] = {}

    def __repr__(self):
        state_map_fn = ""
        if self.state_map_fn is not None:
            state_map_fn = f", {self.state_map_fn!r}"
        step = ""
        if self.step is not None:
            step = f", current_step={self.step!r}"
        unwind = ""
        if self.unwind is not None:
            unwind = f", unwind={self.unwind!r}"
        num_noise_vars = ""
        if self.num_noise_vars:
            num_noise_vars = f", num_noise_vars={self.num_noise_vars!r}"
        return f"TemporalSymbol({self.symbol!r}, {self.network!r}, {self.transition_fn!r}{state_map_fn}{unwind}{num_noise_vars}{step})"

    def __str__(self):
        step = self.step if self.step is not None else "t"
        return f"{self.symbol}_{step}"

    def initial_state(self):
        if "initial_state" not in self._cache:
            _ = self.controller()
        return self._cache["initial_state"]

    def controller(self):
        if "controller" not in self._cache:
            assert self.network.is_concrete
            if self.state_map_fn is not None:
                assert self.state_map_fn.is_concrete
                op_graph = self.network.value.compose(self.state_map_fn.value)
            else:
                op_graph = self.network.value
            op_graph = MLPTransformer().transform(op_graph)

            class GetInputs(OperationVisitor):
                def __init__(self):
                    super().__init__()
                    self.inputs = []

                def walk(self, op_graph: OperationGraph):
                    op_graph.walk(self)
                    return self

                def visit_Input(self, op):
                    self.inputs.append(op)

            inputs = GetInputs().walk(op_graph).inputs
            assert len(inputs) == 1
            initial_state: operations.Input = inputs.pop()
            self._cache["initial_state"] = initial_state
            controller = MLPTransformer().transform(
                OperationGraph(
                    [
                        operations.Concat(
                            (initial_state,) + op_graph.output_operations, axis=1
                        )
                    ]
                )
            )
            self._cache["controller"] = controller
        return self._cache["controller"]

    def stepper(self):
        if "stepper" not in self._cache:
            control_op_graph = self.controller()
            plant_op_graph = self.transition_fn.value
            initial_state = self.initial_state()
            if self.num_noise_vars:
                noise_inputs = (
                    operations.Input(
                        (1, self.num_noise_vars), dtype=initial_state.dtype
                    ),
                )
                plant_input = operations.Concat(
                    control_op_graph.output_operations + noise_inputs,
                    1,
                )
                stepper = plant_op_graph.compose(OperationGraph([plant_input]))
            else:
                stepper = plant_op_graph.compose(control_op_graph)
            self._cache["stepper"] = stepper
            _ = stepper.input_details
            _ = stepper.output_details
        return self._cache["stepper"]

    def freeze(self):
        if not self.step:
            return self.symbol
        if ("result", self.step) in self._cache:
            return self._cache[("result", self.step)]

        initial_state = self.initial_state()
        current_state = OperationGraph([initial_state])
        stepper = self.stepper()

        for i in range(self.step):
            if i in self._cache:
                current_state = self._cache[i]
                continue
            if self.num_noise_vars:
                input_details = current_state.input_details
                output_details = current_state.output_details
                new_op_details = (
                    TensorDetails((1, self.num_noise_vars), initial_state.dtype),
                )
                current_state = OperationGraph(
                    current_state.output_operations
                    + (
                        operations.Input(
                            (1, self.num_noise_vars), dtype=initial_state.dtype
                        ),
                    )
                )
                current_state._input_details = input_details + new_op_details
                current_state._output_details = output_details + new_op_details
            current_state = stepper.compose(current_state).simplify()
            self._cache[i] = current_state
        op_graph = current_state
        if self.num_noise_vars:
            total_noise_var_count = self.unwind * self.num_noise_vars
            step_noise_var_count = self.step * self.num_noise_vars
            op_graph = MLPTransformer().transform(op_graph)
            op_graph = op_graph.compose(
                OperationGraph(
                    [
                        operations.Gemm(
                            operations.Input(
                                (
                                    1,
                                    initial_state.shape[1] + total_noise_var_count,
                                ),
                                initial_state.dtype,
                                name="input",
                            ),
                            np.block(
                                [
                                    [
                                        np.eye(
                                            initial_state.shape[1]
                                            + step_noise_var_count,
                                            dtype=initial_state.dtype,
                                        ),
                                        np.zeros(
                                            (
                                                initial_state.shape[1]
                                                + step_noise_var_count,
                                                total_noise_var_count
                                                - step_noise_var_count,
                                            ),
                                            dtype=initial_state.dtype,
                                        ),
                                    ]
                                ]
                            ).T,
                            np.zeros(
                                initial_state.shape[1] + step_noise_var_count,
                                dtype=initial_state.dtype,
                            ),
                        )
                    ]
                )
            )
        if self.state_map_fn is not None:
            network_steps = "○".join(
                f"({self.transition_fn}○!({self.network}○{self.state_map_fn}))"
                for _ in range(self.step)
            )
        else:
            network_steps = "○".join(
                f"({self.transition_fn}○!{self.network})" for _ in range(self.step)
            )
        network_name = f"({network_steps})"
        result = Network(network_name).concretize(op_graph)(self.symbol)
        self._cache[("result", self.step)] = result
        return result

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
                unwind=self.unwind,
                num_noise_vars=self.num_noise_vars,
                current_step=step + 1,
                ctx=self.ctx,
            )
            next_symbol._cache = self._cache
            return next_symbol
        raise LTLUnwindingBoundReached()


__all__ = ["TemporalSymbol"]
