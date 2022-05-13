"""
dnnv_manage - management tool for DNNV
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import install, list_verifiers, uninstall, verifier_choices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="dnnv_manage - management tool for DNNV",
        prog="dnnv_manage",
    )
    parser.add_argument("-V", "--version", action="version", version="0.0.0")
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show messages with finer-grained information",
        dest="debug",
    )
    verbosity_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress non-essential messages",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        help="commands to manage DNNV",
    )

    install_parser = subparsers.add_parser("install", help="install a verifier")
    install_parser.add_argument(
        "verifiers",
        type=str,
        nargs="+",
        choices=verifier_choices,
        help="the verifier to install",
    )
    install_parser.set_defaults(command=install)

    uninstall_parser = subparsers.add_parser("uninstall", help="uninstall a verifier")
    uninstall_parser.add_argument(
        "verifiers",
        type=str,
        nargs="+",
        choices=verifier_choices,
        help="the verifier to uninstall",
    )
    uninstall_parser.set_defaults(command=uninstall)

    list_parser = subparsers.add_parser("list", help="list installed verifiers")
    list_parser.set_defaults(command=list_verifiers)

    parser.set_defaults(command=lambda *args, **kwargs: 0)
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
    if args.debug or args.quiet:
        args.verbose = False
    return args


def _main() -> int:
    args = parse_args()

    logger = logging.getLogger("dnnv_manage")
    logger.propagate = False
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(levelname)-8s %(asctime)s (%(name)s) %(message)s")
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    params = vars(args).copy()
    params.pop("command")
    params.pop("debug")
    params.pop("verbose")
    params.pop("quiet")
    return args.command(**params)


if __name__ == "__main__":
    _main()
