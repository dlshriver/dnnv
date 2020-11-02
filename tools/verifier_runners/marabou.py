#!/usr/bin/env python
import argparse
import logging
import numpy as np
import os
import sys

sys.path.pop(0)

from pathlib import Path
from maraboupy import Marabou, MarabouCore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("constraints")

    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-p", "--num_workers", "--workers", type=int, default=1)

    return parser.parse_args()


def main(args):
    (A_input, b_input), (A_output, b_output) = np.load(
        args.constraints, allow_pickle=True
    )
    network = Marabou.read_onnx(args.model)

    inputVars = network.inputVars[0]
    outputVars = network.outputVars

    for i, (a, b) in enumerate(zip(A_input, b_input)):
        network.addInequality(list(inputVars.flatten()), list(a), b)

    for i, (a, b) in enumerate(zip(A_output, b_output)):
        network.addInequality(list(outputVars.flatten()), list(a), b)

    options = MarabouCore.Options()
    options._numWorkers = args.num_workers
    result = network.solve(options=options)

    is_unsafe = bool(result[0])
    print("UNSAFE" if is_unsafe else "SAFE")

    if args.output is not None:
        cex = None
        if is_unsafe:
            cex = np.zeros_like(inputVars, dtype=np.float32)
            for flat_index, multi_index in enumerate(np.ndindex(cex.shape)):
                cex[multi_index] = result[0][flat_index]
            print(cex)
        np.save(
            args.output, (is_unsafe, cex),
        )


if __name__ == "__main__":
    main(parse_args())
