#!/usr/bin/env python
import argparse
import logging
import numpy as np
import os
import sys

from pathlib import Path

os.chdir(Path(__file__).parent.parent / "third_party/VeriNet")
sys.path.append(".")

import src.util.logger


def get_logger(level, name: str, directory: str, filename: str):
    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


src.util.logger.get_logger = get_logger

from src.algorithm.verification_objectives import LocalRobustnessObjective
from src.algorithm.verinet import VeriNet
from src.algorithm.verinet_util import Status
from src.data_loader.onnx_parser import ONNXParser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("input_bounds")

    parser.add_argument("-o", "--output", type=str)

    parser.add_argument("-p", "--max_procs", "--procs", type=int, default=1)
    parser.add_argument("-T", "--timeout", type=float, default=24 * 60 * 60)
    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument("--no_split", action="store_true")
    return parser.parse_args()


def main(args):
    onnx_parser = ONNXParser(args.model)
    model = onnx_parser.to_pytorch()

    input_bounds = np.load(args.input_bounds)
    output_bounds = np.array([[0.0, np.finfo(np.float32).max], [0.0, 0.0]])

    solver = VeriNet(model, max_procs=args.max_procs)
    objective = LocalRobustnessObjective(
        correct_class=0, input_bounds=input_bounds, output_bounds=output_bounds
    )

    solver.verify(
        objective, timeout=args.timeout, verbose=args.verbose, no_split=args.no_split
    )

    if args.output is not None:
        np.save(
            args.output,
            (str(solver.status).split(".")[-1], solver.counter_example),
        )
    print(str(solver.status).split(".")[-1])
    if solver.counter_example is not None:
        print(solver.counter_example)

    print("")
    print(f"Result: {solver.status}")
    print(f"Branches explored: {solver.branches_explored}")
    print(f"Maximum depth reached: {solver.max_depth}")


if __name__ == "__main__":
    main(parse_args())
