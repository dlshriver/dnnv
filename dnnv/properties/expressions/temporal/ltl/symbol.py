from __future__ import annotations

from typing import Any, Dict, Optional, Union

from .....nn import OperationGraph, operations
from .....nn.transformers.mlp import MLPTransformer, SqueezeGemms
from ...base import Expression
from ...terms import Network, Symbol
from .base import LTL, LTLMaxStepReached

IntExpression = Union[int, Expression]


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
        **kwargs,
    ):
        """An expression representing a temporal symbol.

        At each step, this expression represents the current value of `symbol`,
        whose value is obtained by repeated applications of `state_map_fn`,
        `network`, and `transition_fn`.
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
            op_graph: OperationGraph = self.network.value
            if self.state_map_fn is not None:
                assert self.state_map_fn.is_concrete
                op_graph = op_graph.compose(self.state_map_fn.value)

            current_state = operations.Concat(
                op_graph.inputs + op_graph.outputs, axis=1
            )
            op_graph = OperationGraph([current_state])
            op_graph = MLPTransformer().transform(op_graph, keep_generators=True)
            self._cache["controller"] = op_graph
        return self._cache["controller"]

    def plant(self):
        if "plant" not in self._cache:
            if self.control_period > 1:
                # TODO
                assert 0, "TODO"
            assert self.transition_fn.is_concrete
            op_graph: OperationGraph = self.transition_fn.value
            input_details = op_graph.input_details
            output_details = op_graph.output_details
            assert len(input_details) == 1
            assert len(output_details) == 1
            op_graph = MLPTransformer().transform(op_graph, keep_generators=True)
            self._cache["plant"] = op_graph
        return self._cache["plant"]

    def freeze(self):
        if not self.control_step and not self.plant_step:
            return self.symbol
        if ("frozen", self.control_step, self.plant_step) in self._cache:
            return self._cache[("frozen", self.control_step, self.plant_step)]

        controller = self.controller()
        plant = self.plant()

        _ = self.previous().freeze()
        if self.control_step == 1 and self.plant_step == 0:
            if len(controller.inputs) > 1:
                current_state: OperationGraph = OperationGraph(
                    [operations.Concat(controller.inputs, axis=1)]
                )
            else:
                current_state = OperationGraph([controller.inputs[0]])
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
            current_state = self._cache[
                ("state", self.control_step - 1, self.control_period - 1)
            ]
            control_graph = controller.compose(current_state)
            control_signal = control_graph
        state_graph = plant.compose(control_signal)
        state_graph = SqueezeGemms().transform(state_graph)
        current_state = MLPTransformer().transform(
            state_graph, keep_generators=True, _pre_simplified=True
        )

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

    def previous(self):
        if isinstance(self.control_period, Expression):
            assert self.control_period.is_concrete
            self.control_period = self.control_period.value
        control_step = self.control_step or 0
        plant_step = self.plant_step
        if plant_step is None:
            plant_step = self.control_period - 1
        self._simplify_network_expressions()
        if control_step * self.control_period + plant_step > 0:
            prev_plant_step = (plant_step - 1) % self.control_period
            prev_control_step = control_step
            if plant_step == 0:
                prev_control_step = control_step - 1
            prev_symbol = TemporalSymbol(
                self.symbol,
                self.network,
                self.transition_fn,
                self.state_map_fn,
                max_step=self.max_step,
                control_step=prev_control_step,
                plant_step=prev_plant_step,
                control_period=self.control_period,
                ctx=self.ctx,
            )
            prev_symbol._cache = self._cache
            return prev_symbol
        raise RuntimeError(
            "Cannot get previous step of initial conditions."
        )  # TODO : use a custom exception type

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
        if isinstance(self.control_period, Expression) or self.control_period > 1:
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
