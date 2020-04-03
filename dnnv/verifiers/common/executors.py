import multiprocessing as mp
import select
import subprocess as sp

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from dnnv import logging

from .errors import VerifierError


class VerifierExecutor(ABC):
    @abstractmethod
    def run(self):
        raise NotImplementedError()


class CommandLineExecutor(VerifierExecutor):
    """Executes verifiers using their command line interface.
    """

    def __init__(self, *args: str, verifier_error: Type[VerifierError] = VerifierError):
        self.args = ("stdbuf", "-oL", "-eL") + args
        self.verifier_error = verifier_error
        self.output_lines = []  # type: List[str]
        self.error_lines = []  # type: List[str]

    def run(self):
        logger = logging.getLogger(__name__)
        arg_string = " ".join(self.args)
        logger.info(f"EXECUTING: {arg_string}")
        proc = None
        try:
            proc = sp.Popen(self.args, stdout=sp.PIPE, stderr=sp.PIPE, encoding="utf8")

            self.output_lines = []
            self.error_lines = []
            while proc.poll() is None:
                for (name, stream, lines) in [
                    ("STDOUT", proc.stdout, self.output_lines),
                    ("STDERR", proc.stderr, self.error_lines),
                ]:
                    ready, _, _ = select.select([stream], [], [], 0)
                    if not ready:
                        continue
                    line = stream.readline()
                    if not line:
                        continue
                    line = line.strip()
                    lines.append(line)
                    logger.debug(f"[{name}]:{line}")
            for line in proc.stdout.readlines():
                line = line.strip()
                logger.debug(f"[STDOUT]:{line}")
                self.output_lines.append(line)
            for line in proc.stderr.readlines():
                line = line.strip()
                logger.debug(f"[STDERR]:{line}")
                self.error_lines.append(line)
            if proc.returncode != 0:
                raise self.verifier_error(f"Return code: {proc.returncode}")
        finally:
            if proc is not None:
                proc.stderr.close()
                proc.stdout.close()

        return self.output_lines, self.error_lines
