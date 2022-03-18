from __future__ import annotations

import itertools
import numpy as np

from typing import Any, Dict, Optional, Union

from .base import LTL, LTLMaxStepReached
from ...base import Expression
from ...terms import Constant, Network, Parameter, Symbol
from .....nn import operations, OperationGraph, OperationVisitor
from .....nn.transformers.mlp import MLPTransformer
from .....nn.utils import TensorDetails

IntExpression = Union[int, Expression]


def _concat_opgraphs(*op_graphs: OperationGraph):
    input_set = set()
    inputs = []
    outputs = []
    input_details = []
    output_details = []
    for graph in op_graphs:
        for input_op, details in zip(graph.inputs, graph.input_details):
            if input_op not in input_set:
                inputs.append(input_op)
                input_details.append(details)
                input_set.add(input_op)
        outputs += graph.outputs
        output_details += graph.output_details
    concatenated_op_graph = OperationGraph(
        [
            operations.Concat(
                outputs,
                axis=1,
            )
        ]
    )
    concatenated_op_graph._inputs = inputs
    concatenated_op_graph._input_details = input_details
    details = output_details[0]
    shape = list(details.shape)
    dtype = details.dtype
    for o in output_details[1:]:
        assert dtype == o.dtype
        shape[1] += o.shape[1]
    concatenated_op_graph._output_details = [TensorDetails(tuple(shape), dtype)]
    return concatenated_op_graph


class TemporalSymbol(LTL):
    def __init__(
        self,
        symbol: Symbol,
        network: Network,
        transition_fn: Network,
        state_map_fn: Optional[Network] = None,
        *,
        max_step: Optional[IntExpression] = None,
        control_step: Optional[IntExpression] = None,
        plant_step: Optional[IntExpression] = None,
        control_period: IntExpression = 1,
        state_slack: Optional[Any] = None,
        state_bounds: Optional[Any] = None,
        **kwargs,
    ):
        """TODO

        TODO
        """
        super().__init__(**kwargs)
        self.symbol = symbol
        self.network = network
        self.transition_fn = transition_fn
        self.state_map_fn = state_map_fn

        assert max_step is not None, "parameter `max_step` is required"
        if isinstance(max_step, Expression) and max_step.is_concrete:
            max_step = max_step.value
        self.max_step = max_step

        if (
            control_step is not None
            and isinstance(control_step, Expression)
            and control_step.is_concrete
        ):
            control_step = control_step.value
        self.control_step = control_step

        if (
            plant_step is not None
            and isinstance(plant_step, Expression)
            and plant_step.is_concrete
        ):
            plant_step = plant_step.value
        self.plant_step = plant_step

        if isinstance(control_period, Expression) and control_period.is_concrete:
            control_period = control_period.value
        self.control_period = control_period

        self._cache: Dict[Any, Any] = {}

    def controller(self):
        if "controller" not in self._cache:
            assert self.network.is_concrete
            if self.state_map_fn is not None:
                assert self.state_map_fn.is_concrete
                op_graph: OperationGraph = self.network.value.compose(
                    self.state_map_fn.value
                )
            else:
                op_graph = self.network.value
            op_graph = MLPTransformer().transform(op_graph)
            _ = op_graph.output_details
            self._cache["controller"] = op_graph
        return self._cache["controller"]

    def plant(self):
        if "plant" not in self._cache:
            assert self.transition_fn.is_concrete
            op_graph: OperationGraph = self.transition_fn.value
            op_graph = MLPTransformer().transform(op_graph)
            _ = op_graph.output_details
            self._cache["plant"] = op_graph
        return self._cache["plant"]

    def freeze(self):
        if not self.control_step and not self.plant_step:
            return self.symbol
        if ("frozen", self.control_step, self.plant_step) in self._cache:
            return self._cache[("frozen", self.control_step, self.plant_step)]

        controller = self.controller()
        plant = self.plant()

        if self.control_step == 1 and self.plant_step == 0:
            current_state: OperationGraph = OperationGraph(
                [operations.Concat(controller.inputs, axis=1)]
            )
            _ = current_state.output_details
            self._cache[
                ("state", self.control_step - 1, self.control_period - 1)
            ] = current_state

        if self.plant_step > 0:
            control_signal: OperationGraph = self._cache[
                ("control_signal", self.control_step)
            ]
            current_state = self._cache[
                ("state", self.control_step, self.plant_step - 1)
            ]
        else:
            current_state = current_state = self._cache[
                ("state", self.control_step - 1, self.control_period - 1)
            ]
            _ = current_state.inputs
            _ = current_state.output_details
            control_graph = controller.compose(current_state)
            control_signal = control_graph
        current_state_and_control_graph = _concat_opgraphs(
            current_state, control_signal
        )
        state_graph = plant.compose(current_state_and_control_graph)
        current_state = MLPTransformer().transform(state_graph)

        self._cache[("control_signal", self.control_step)] = control_signal
        self._cache[("state", self.control_step, self.plant_step)] = current_state

        network = Network(self._get_frozen_network_name())
        network.concretize(current_state)
        self._cache[("frozen", self.control_step, self.plant_step)] = network(
            self.symbol
        )
        return network(self.symbol)

    def next(self):
        if isinstance(self.control_period, Expression):
            assert self.control_period.is_concrete
            self.control_period = self.control_period.value
        control_step = self.control_step or 0
        plant_step = self.plant_step
        if plant_step is None:
            plant_step = self.control_period - 1
        self._simplify_network_expressions()
        if (
            control_step * self.control_period + plant_step
            < self.max_step * self.control_period + self.control_period - 1
        ):
            next_plant_step = (plant_step + 1) % self.control_period
            next_control_step = control_step
            if next_plant_step == 0:
                next_control_step = control_step + 1
            next_symbol = TemporalSymbol(
                self.symbol,
                self.network,
                self.transition_fn,
                self.state_map_fn,
                max_step=self.max_step,
                control_step=next_control_step,
                plant_step=next_plant_step,
                control_period=self.control_period,
                ctx=self.ctx,
            )
            next_symbol._cache = self._cache
            return next_symbol
        raise LTLMaxStepReached()

    def _simplify_network_expressions(self):
        if self.network.is_concrete and not isinstance(self.network, Network):
            self.network = self.network.propagate_constants()
        if self.transition_fn.is_concrete and not isinstance(
            self.transition_fn, Network
        ):
            self.transition_fn = self.transition_fn.propagate_constants()
        if (
            self.state_map_fn is not None
            and self.state_map_fn.is_concrete
            and not isinstance(self.state_map_fn, Network)
        ):
            self.state_map_fn = self.state_map_fn.propagate_constants()

    def __repr__(self):
        state_map_fn = ""
        if self.state_map_fn is not None:
            state_map_fn = f", {self.state_map_fn!r}"
        control_step = ""
        if self.control_step is not None:
            control_step = f", control_step={self.control_step!r}"
        plant_step = ""
        if self.plant_step is not None:
            plant_step = f", plant_step={self.plant_step!r}"
        max_step = ""
        if self.max_step is not None:
            max_step = f", max_step={self.max_step!r}"
        control_period = ""
        if self.control_period > 1:
            control_period = f", control_period={self.control_period!r}"
        return f"TemporalSymbol({self.symbol!r}, {self.network!r}, {self.transition_fn!r}{state_map_fn}{max_step}{control_step}{plant_step}{control_period})"

    def __str__(self):
        step = self.control_step if self.control_step is not None else "t"
        return f"{self.symbol}_{step}"

    def _get_frozen_network_name(self):
        if self.state_map_fn is not None:
            controller_name = f"({self.network}○{self.state_map_fn})'"
        else:
            controller_name = f"{self.network}'"
        plant_name = f"{self.transition_fn}'"
        network_name = controller_name
        for _ in range(self.control_step - 1):
            for _ in range(self.control_period):
                network_name = f"{plant_name}○({network_name})"
            network_name = f"{controller_name}○({network_name})"
        for _ in range(self.plant_step + 1):
            network_name = f"{plant_name}○({network_name})"
        return f"({network_name})"


__all__ = ["TemporalSymbol"]
