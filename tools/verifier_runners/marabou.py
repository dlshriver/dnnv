#!/usr/bin/env python
import argparse
import logging
import numpy as np
import os
import sys

sys.path.pop(0)

from pathlib import Path
from maraboupy import Marabou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("constraints")

    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-p", "--num_processes", "--procs", type=int, default=1)

    return parser.parse_args()


def main(args):

    lb, ub, (A, b) = np.load(args.constraints, allow_pickle=True)
    network = Marabou.read_onnx(args.model)

    inputVars = network.inputVars[0]
    outputVars = network.outputVars.flatten()

    for index in np.ndindex(network.inputVars[0].shape):
        network.setLowerBound(network.inputVars[0][index], lb[index])
        network.setUpperBound(network.inputVars[0][index], ub[index])

    for i, constraint in enumerate(A):
        network.addInequality(list(outputVars), list(constraint), b[i])

    result = network.solve()

    if args.output is not None:
        is_unsafe = bool(result[0])
        print("UNSAFE" if is_unsafe else "SAFE")
        cex = None
        if is_unsafe:
            cex = np.zeros_like(lb)
            for flat_index, multi_index in enumerate(np.ndindex(cex.shape)):
                cex[multi_index] = result[0][flat_index]
            print(cex)
        np.save(
            args.output, (is_unsafe, cex),
        )

    return


if __name__ == "__main__":
    main(parse_args())
