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
        self.args = args
        self.verifier_error = verifier_error
        self.output_lines = []  # type: List[str]
        self.error_lines = []  # type: List[str]

    def run(self):
        logger = logging.getLogger(__name__)
        arg_string = " ".join(self.args)
        logger.info(f"EXECUTING: {arg_string}")
        try:
            proc = sp.Popen(self.args, stdout=sp.PIPE, stderr=sp.PIPE, encoding="utf8")

            self.output_lines = []
            self.error_lines = []
            while proc.poll() is None:
                readables, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0)
                while len(readables) > 0:
                    io = readables.pop()
                    line = io.readline().strip()
                    if not line:
                        continue
                    prefix = ""
                    if io.fileno() == proc.stdout.fileno():
                        self.output_lines.append(line)
                        prefix = "[STDOUT]:"
                    elif io.fileno() == proc.stderr.fileno():
                        self.error_lines.append(line)
                        prefix = "[STDERR]:"
                    logger.debug(f"{prefix}{line}")
            if proc.returncode != 0:
                raise self.verifier_error(f"Received signal: {-proc.returncode}")

            for line in proc.stderr.readlines():
                line = line.strip()
                if line:
                    logger.debug(f"[STDERR]:{line}")
                    self.error_lines.append(line)
            for line in proc.stdout.readlines():
                line = line.strip()
                if line:
                    logger.debug(f"[STDOUT]:{line}")
                    self.output_lines.append(line)
        finally:
            proc.stderr.close()
            proc.stdout.close()

        return self.output_lines, self.error_lines
