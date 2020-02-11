from typing import List, Optional, Type

from dnnv.nn import OperationGraph
from dnnv.nn.layers import Layer, InputLayer, FullyConnected, Convolutional

from .errors import VerifierError, VerifierTranslatorError


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


def sandbox(target, q, *args, **kwargs):
    result = target(*args, **kwargs)
    q.put(result)
    q.close()


def sandboxed(target=None, verifier_error=VerifierError):
    import functools
    import multiprocessing as mp
    import sys

    if mp.current_process().name == "MainProcess":

        def sandboxer(target):
            @functools.wraps(target)
            def sandboxed_target(*args, **kwargs):
                ctx = mp.get_context("spawn")
                q = ctx.Queue()
                new_target = sys.modules[target.__module__].__dict__[target.__name__]
                p = ctx.Process(
                    target=sandbox,
                    args=(new_target, q) + args,
                    kwargs=kwargs,
                    daemon=True,
                )
                p.start()
                while p.is_alive() and p.exitcode is None and q.empty():
                    pass
                if q.empty():
                    raise verifier_error(f"Verifier exited with status {p.exitcode}")
                result = q.get_nowait()
                p.join()
                p.close()
                return result

            return sandboxed_target

        if target is not None:
            return sandboxer(target)
        return sandboxer
    if target is not None:
        return target
    return lambda x: x
