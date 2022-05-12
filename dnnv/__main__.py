"""
"""
import argparse
import os
import time
import typing
from pathlib import Path
from typing import Collection, List, Optional, Type

import numpy as np

from . import cli, errors
from . import logging_utils as logging
from . import nn, properties, utils
from .verifiers.common.results import VerificationResult

if typing.TYPE_CHECKING:
    from dnnv.verifiers.common.base import Verifier


def main(args: argparse.Namespace, extra_args: Optional[List[str]] = None):
    logger = logging.initialize(__package__, args)
    utils.set_random_seed(args.seed)

    if args.property is not None:
        logger.debug("Reading property %s", args.property)
        phi = properties.parse(
            args.property, format=args.prop_format, args=extra_args
        ).propagate_constants()
        print("Verifying property:")
        print(phi)
        print()
        if extra_args is not None and len(extra_args) > 0:
            logger.error("Unused arguments: %r", extra_args)
            unknown_args = " ".join(extra_args)
            print(f"ERROR: Unknown arguments: {unknown_args}")
            return 1

    if args.networks:
        print("Verifying Networks:")
        networks = {}
        for name, network in args.networks.items():
            print(f"{name}:")
            logger.debug("Parsing network (%s)", network)
            dnn = nn.parse(network)
            dnn.pprint()
            networks[name] = dnn
            print()

        if args.property is None:
            phi = properties.Forall(
                properties.Symbol("*"),
                properties.And(
                    *[
                        properties.Implies(
                            properties.Network(name)(properties.Symbol("*")) > 1,
                            properties.Network(name)(properties.Symbol("*"))
                            < 2 * properties.Network(name)(properties.Symbol("*")),
                        )
                        for name in networks
                    ]
                ),
            )
        phi.concretize(**networks)

    if len(args.verifiers) > 1:
        verifier_names = [v.__module__ for v in args.verifiers]
        logger.error("More than 1 verifier specified: %r", verifier_names)
        print(f"ERROR: More than 1 verifier specified: {verifier_names}")
        return 1
    elif len(args.verifiers) == 0:
        return 0

    verifier: Type[Verifier] = args.verifiers[0]
    verifier_name = verifier.__name__.lower()
    start_t = time.time()
    params = args.verifier_parameters.get(verifier_name, {})
    params["eager_exit"] = not args.no_eager_exit
    try:
        logger.info("verifying property with %s", verifier.__name__.lower())
        verification_result = verifier.verify(phi, **params)
        result_str = str(verification_result.result)
    except errors.DNNVError as e:
        verification_result = VerificationResult()
        result_str = f"{type(e).__name__}({e})"
        logger.debug("Error traceback:", exc_info=True)
    except SystemExit:
        if verifier.__module__ != "dnnv.verifiers.convert":
            logger.error(f"Verifier {verifier_name} called exit()")
            raise
        return 0
    end_t = time.time()
    cex = verification_result.cex
    vtimes: Optional[Collection[float]] = verification_result.vtimes
    if args.save_violation is not None and cex is not None:
        np.save(args.save_violation, cex)
    print(f"{verifier.__module__}")
    print(f"  result: {result_str}")
    print(f"  total time: {(end_t - start_t):.4f}")
    print(f"  verification stats:")
    print(f"    num solved subproblems: {len(vtimes)}")
    print(f"    total time            : {sum(vtimes):.4f}")
    if len(vtimes) > 0:
        print(f"    mean time             : {sum(vtimes)/len(vtimes):.4f}")
        print(f"    min time              : {min(vtimes):.4f}")
        print(f"    max time              : {max(vtimes):.4f}")
    return 0


def _main():
    os.environ["PATH"] = ":".join(
        [
            os.environ.get("PATH", ""),
            str(
                (
                    Path(
                        os.path.join(
                            os.getenv("XDG_DATA_HOME", "~/.local/share"), "dnnv"
                        )
                    )
                    / "bin"
                )
                .expanduser()
                .resolve()
            ),
        ]
    )
    # TODO : only modify path if not in VIRTUAL_ENV
    try:
        return exit(main(*cli.parse_args()))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    exit(_main())
