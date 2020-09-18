"""
"""
import argparse
import time

from typing import List, Optional

from . import cli
from . import logging
from . import nn
from . import properties
from . import utils
from .verifiers.common import VerifierError, VerifierTranslatorError


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
        raise ValueError(f"Unknown arguments: {unknown_args}")

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

    for verifier in args.verifiers:
        start_t = time.time()
        try:
            params = args.verifier_parameters.get(verifier.__name__.lower(), {})
            result = verifier.verify(phi, **params)
        except VerifierTranslatorError as e:
            result = f"{type(e).__name__}({e})"
            logger.debug("Translation Error traceback:", exc_info=True)
        except VerifierError as e:
            result = f"{type(e).__name__}({e})"
            logger.debug("Verifier Error traceback:", exc_info=True)
        end_t = time.time()
        print(f"{verifier.__module__}")
        print(f"  result: {result}")
        print(f"  time: {(end_t - start_t):.4f}")


def _main():
    main(*cli.parse_args())


if __name__ == "__main__":
    _main()
