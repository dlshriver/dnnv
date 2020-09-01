from typing import List, Optional, Type

from dnnv.nn import OperationGraph
from dnnv.nn.layers import Layer, InputLayer, FullyConnected, Convolutional

from .errors import VerifierTranslatorError


def as_layers(
    op_graph: OperationGraph,
    layer_types: Optional[List[Type[Layer]]] = None,
    extra_layer_types: Optional[List[Type[Layer]]] = None,
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
) -> List[Layer]:
    if layer_types is None:
        layer_types = [InputLayer, FullyConnected, Convolutional]
    if extra_layer_types is not None:
        layer_types = layer_types + extra_layer_types
    layers: List[Layer] = []
    while True:
        layer_match = Layer.match(op_graph, layer_types=layer_types)
        if layer_match is None:
            break
        layers.insert(0, layer_match.layer)
        op_graph = layer_match.input_op_graph
    if len(op_graph.output_operations) > 0:
        raise translator_error("Unsupported computation graph detected")
    return layers


__all__ = ["as_layers"]
