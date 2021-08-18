#!/usr/bin/env python
import argparse
import numpy as np
import sys

sys.path.pop(0)

from pathlib import Path
from nnenum.enumerate import enumerate_network
from nnenum.lp_star import LpStar
from nnenum.onnx_network import load_onnx_network
from nnenum.settings import Settings
from nnenum.specification import Specification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("constraints")
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-p", "--num_processes", "--procs", type=int)
    return parser.parse_args()


def main(args):
    Settings.UNDERFLOW_BEHAVIOR = "warn"
    Settings.PRINT_PROGRESS = False
    if args.num_processes is not None:
        Settings.NUM_PROCESSES = args.num_processes
    (lb, ub), (A_input, b_input), (A_output, b_output) = np.load(
        args.constraints, allow_pickle=True
    )
    network = load_onnx_network(args.model)
    ninputs = A_input.shape[1]
    init_box = np.array(
        list(
            zip(
                lb.reshape(network.get_input_shape()).flatten("F"),
                ub.reshape(network.get_input_shape()).flatten("F"),
            )
        ),
        dtype=np.float32,
    )
    init_star = LpStar(
        np.eye(ninputs, dtype=np.float32), np.zeros(ninputs, dtype=np.float32), init_box
    )
    for a, b in zip(A_input, b_input):
        a_ = a.reshape(network.get_input_shape()).flatten("F")
        init_star.lpi.add_dense_row(a_, b)
    spec = Specification(A_output, b_output)
    result = enumerate_network(init_star, network, spec)
    print(result.result_str)
    if args.output is not None:
        cex = None
        if result.cinput is not None:
            cex = (
                np.array(list(result.cinput))
                .astype(np.float32)
                .reshape(network.get_input_shape(), order="F")
            )
            print(cex)
        np.save(args.output, (result.result_str, cex))
    return


if __name__ == "__main__":
    main(parse_args())
