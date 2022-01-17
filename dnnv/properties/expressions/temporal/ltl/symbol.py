from __future__ import annotations

import numpy as np

from typing import Any, Dict, Optional, Union

from .base import LTL, LTLUnwindingBoundReached
from ...terms import Constant, Network, Parameter, Symbol
from .....nn import operations, OperationGraph, OperationVisitor
from .....nn.transformers.mlp import MLPTransformer
from .....nn.utils import TensorDetails


class GetInputs(OperationVisitor):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def walk(self, op_graph: OperationGraph):
        op_graph.walk(self)
        return self

    def visit_Input(self, op):
        self.inputs.append(op)


class TemporalSymbol(LTL):
    def __init__(
        self,
        symbol: Symbol,
        network: Network,
        transition_fn: Network,
        state_map_fn: Optional[Network] = None,
        *,
        unwind: Optional[Union[int, Constant, Parameter]] = None,
        control_period: Union[int, Constant, Parameter] = 1,
        num_noise_vars: Union[int, Constant, Parameter] = 0,
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
            isinstance(control_period, (Constant, Parameter))
            and control_period.is_concrete
        ):
            control_period = control_period.value
        self.control_period = control_period

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

    def state_details(self):
        if "state_details" not in self._cache:
            if self.state_map_fn is not None:
                op_graph: OperationGraph = self.state_map_fn.value
            else:
                self.network = self.network.propagate_constants()
                op_graph = self.network.value
            state_details = op_graph.input_details
            assert len(state_details) == 1
            self._cache["state_details"] = (
                np.product(state_details[0].shape[1:]),
                state_details[0].dtype,
            )
        return self._cache["state_details"]

    def control_details(self):
        if "control_details" not in self._cache:
            self.network = self.network.propagate_constants()
            op_graph: OperationGraph = self.network.value
            control_details = op_graph.output_details
            assert len(control_details) == 1
            self._cache["control_details"] = (
                np.product(control_details[0].shape[1:]),
                control_details[0].dtype,
            )
        return self._cache["control_details"]

    def initial_state(self):
        if "initial_state" not in self._cache:
            num_state_variables, dtype = self.state_details()
            initial_state = operations.Input((1, num_state_variables), dtype)
            if self.control_period > 1:
                num_control_vars, dtype = self.control_details()
                initial_state = operations.Gemm(
                    initial_state,
                    np.block(
                        [
                            [
                                np.eye(num_state_variables, dtype=dtype),
                                np.zeros(
                                    (num_state_variables, num_control_vars), dtype=dtype
                                ),
                            ]
                        ]
                    ),
                    np.zeros(num_state_variables + num_control_vars, dtype=dtype),
                )
            self._cache["initial_state"] = initial_state
        return self._cache["initial_state"]

    def controller(self):
        if "controller" not in self._cache:
            assert self.network.is_concrete
            self.network = self.network.propagate_constants()
            if self.state_map_fn is not None:
                assert self.state_map_fn.is_concrete
                self.state_map_fn = self.state_map_fn.propagate_constants()
                op_graph = self.network.value.compose(self.state_map_fn.value)
            else:
                op_graph = self.network.value
            inputs = GetInputs().walk(op_graph).inputs
            assert len(inputs) == 1
            initial_state: operations.Input = inputs.pop()
            op_graph = OperationGraph(
                [
                    operations.Concat(
                        (initial_state,) + op_graph.output_operations, axis=1
                    )
                ]
            )
            op_graph = MLPTransformer().transform(op_graph)
            if self.control_period > 1:
                state_control_details = op_graph.output_details
                assert len(state_control_details) == 1
                assert len(state_control_details[0].shape) == 2
                num_state_variables, state_dtype = self.state_details()
                num_control_signals, control_dtype = self.control_details()
                shape = (1, num_state_variables + num_control_signals)
                dtype = state_dtype
                op_graph = op_graph.compose(
                    OperationGraph(
                        [
                            operations.Gemm(
                                operations.Input(shape, dtype),
                                np.block(
                                    [
                                        [
                                            np.eye(num_state_variables, dtype=dtype),
                                            np.zeros(
                                                (
                                                    num_state_variables,
                                                    num_control_signals,
                                                ),
                                                dtype=dtype,
                                            ),
                                        ]
                                    ]
                                ).T,
                                np.zeros(num_state_variables, dtype=dtype),
                            )
                        ]
                    )
                )
            controller = MLPTransformer().transform(op_graph)
            self._cache["controller"] = controller
        return self._cache["controller"]

    def plant(self):
        if "plant" not in self._cache:
            self.transition_fn = self.transition_fn.propagate_constants()
            plant_op_graph = self.transition_fn.value
            if self.control_period > 1:
                plant_inputs = GetInputs().walk(plant_op_graph).inputs
                assert len(plant_inputs) == 1
                num_state_variables, _ = self.state_details()
                num_control_signals, dtype = self.control_details()
                plant_output_ops = plant_op_graph.copy().output_operations + (
                    operations.Gemm(
                        plant_inputs[0],
                        np.block(
                            [
                                [
                                    np.zeros(
                                        (num_control_signals, num_state_variables),
                                        dtype=dtype,
                                    ),
                                    np.eye(num_control_signals, dtype=dtype),
                                    np.zeros(
                                        (num_control_signals, self.num_noise_vars),
                                        dtype=dtype,
                                    ),
                                ]
                            ]
                        ).T,
                        np.zeros(num_control_signals, dtype=dtype),
                    ),
                )
                plant_op_graph = MLPTransformer().transform(
                    OperationGraph(plant_output_ops)
                )
            self._cache["plant"] = plant_op_graph
        return self._cache["plant"]

    def stepper(self):
        if "stepper" not in self._cache:
            control_op_graph = self.controller()
            plant_op_graph = self.plant()
            if self.num_noise_vars:
                _, dtype = self.state_details()
                noise_inputs = (
                    operations.Input((1, self.num_noise_vars), dtype=dtype),
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
        plant = self.plant()
        stepper = self.stepper()
        num_state_variables, state_dtype = self.state_details()

        for step in range(self.step):
            if step in self._cache:
                current_state = self._cache[step]
                continue
            if self.num_noise_vars:
                input_details = current_state.input_details
                output_details = current_state.output_details
                new_op_details = (TensorDetails((1, self.num_noise_vars), state_dtype),)
                current_state = OperationGraph(
                    current_state.output_operations
                    + (operations.Input((1, self.num_noise_vars), dtype=state_dtype),)
                )
                current_state._input_details = input_details + new_op_details
                current_state._output_details = output_details + new_op_details
            if (step % self.control_period) == 0:
                current_state = stepper.compose(current_state).simplify()
            else:
                if self.num_noise_vars:
                    current_state = MLPTransformer().transform(current_state)
                current_state = plant.compose(current_state).simplify()
            self._cache[step] = current_state
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
                                    num_state_variables + total_noise_var_count,
                                ),
                                state_dtype,
                                name="input",
                            ),
                            np.block(
                                [
                                    [
                                        np.eye(
                                            num_state_variables + step_noise_var_count,
                                            dtype=state_dtype,
                                        ),
                                        np.zeros(
                                            (
                                                num_state_variables
                                                + step_noise_var_count,
                                                total_noise_var_count
                                                - step_noise_var_count,
                                            ),
                                            dtype=state_dtype,
                                        ),
                                    ]
                                ]
                            ).T,
                            np.zeros(
                                num_state_variables + step_noise_var_count,
                                dtype=state_dtype,
                            ),
                        )
                    ]
                )
            )
        if self.control_period > 1:
            num_control_signals, _ = self.control_details()
            op_graph = OperationGraph(
                [
                    operations.Gemm(
                        operations.Input(
                            (
                                1,
                                num_state_variables + num_control_signals,
                            ),
                            state_dtype,
                            name="input",
                        ),
                        np.block(
                            [
                                [
                                    np.eye(
                                        num_state_variables,
                                        dtype=state_dtype,
                                    ),
                                    np.zeros(
                                        (
                                            num_state_variables,
                                            num_control_signals,
                                        ),
                                        dtype=state_dtype,
                                    ),
                                ]
                            ]
                        ).T,
                        np.zeros(
                            num_state_variables,
                            dtype=state_dtype,
                        ),
                    )
                ]
            ).compose(op_graph)
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
                control_period=self.control_period,
                num_noise_vars=self.num_noise_vars,
                current_step=step + 1,
                ctx=self.ctx,
            )
            next_symbol._cache = self._cache
            return next_symbol
        raise LTLUnwindingBoundReached()


__all__ = ["TemporalSymbol"]
