"""
"""
import argparse
import numpy as np
import time

from typing import List, Optional

from . import cli
from . import logging
from . import nn
from . import properties
from . import utils
from .verifiers.common import VerifierError, VerifierTranslatorError, SAT


def main(args: argparse.Namespace, extra_args: Optional[List[str]] = None):
    logger = logging.initialize(__package__, args)
    utils.set_random_seed(args.seed)

    logger.debug("Reading property %s", args.property)
    phi = properties.parse(args.property, format=args.prop_format, args=extra_args)
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
            dnn = nn.parse(network).simplify()
            dnn.pprint()
            networks[name] = dnn
            print()

        phi.concretize(**networks)

    if len(args.verifiers) > 1:
        verifier_names = [v.__module__ for v in args.verifiers]
        logger.error("More than 1 verifier specified: %r", verifier_names)
        print(f"ERROR: More than 1 verifier specified: {verifier_names}")
        return 1
    elif len(args.verifiers) == 0:
        return 0

    verifier = args.verifiers[0]
    verifier_name = verifier.__name__.lower()
    start_t = time.time()
    try:
        params = args.verifier_parameters.get(verifier_name, {})
        result, cex = verifier.verify(phi, **params)
    except VerifierTranslatorError as e:
        result = f"{type(e).__name__}({e})"
        logger.debug("Translation Error traceback:", exc_info=True)
    except VerifierError as e:
        result = f"{type(e).__name__}({e})"
        logger.debug("Verifier Error traceback:", exc_info=True)
    except SystemExit:
        if verifier.__module__ != "dnnv.verifiers.convert":
            logger.error(f"Verifier {verifier_name} called exit()")
            raise
        return 0
    end_t = time.time()
    if result == SAT and args.save_violation is not None and cex is not None:
        np.save(args.save_violation, cex)
    print(f"{verifier.__module__}")
    print(f"  result: {result}")
    print(f"  time: {(end_t - start_t):.4f}")
    return 0


def _main():
    return exit(main(*cli.parse_args()))


if __name__ == "__main__":
    exit(_main())
