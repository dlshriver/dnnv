#!/usr/bin/env python
import argparse
import logging
import numpy as np
import os
import sys

sys.path.pop(0)

from pathlib import Path

from nnenum.enumerate import enumerate_network
from nnenum.onnx_network import load_onnx_network
from nnenum.settings import Settings
from nnenum.specification import Specification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("constraints")

    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-p", "--num_processes", "--procs", type=int, default=1)

    return parser.parse_args()


def main(args):
    Settings.PRINT_PROGRESS = False
    Settings.NUM_PROCESSES = args.num_processes

    init_box, A, b = np.load(args.constraints, allow_pickle=True)
    network = load_onnx_network(args.model)

    spec = Specification(A, b)

    result = enumerate_network(init_box, network, spec)

    if args.output is not None:
        cex = np.array(list(result.cinput)) if result.cinput is not None else None
        np.save(
            args.output,
            (result.result_str, cex),
        )

    print(result.result_str)

    return


if __name__ == "__main__":
    main(parse_args())
