import argparse
import logging
import psutil
import signal
import subprocess as sp
import sys
import time


def memory_t(value):
    if isinstance(value, int):
        return value
    elif value.lower().endswith("g"):
        return int(value[:-1]) * 1_000_000_000
    elif value.lower().endswith("m"):
        return int(value[:-1]) * 1_000_000
    elif value.lower().endswith("k"):
        return int(value[:-1]) * 1000
    else:
        return int(value)


def memory_repr(value):
    if value > 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}G"
    elif value > 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value > 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value}"


def _parse_args():
    parser = argparse.ArgumentParser(description="resmonitor - monitor resource usage")

    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show messages with finer-grained information",
    )
    verbosity_group.add_argument(
        "-q", "--quiet", action="store_true", help="suppress non-essential messages"
    )

    parser.add_argument(
        "-T", "--time", default=-1, type=float, help="The max running time in seconds."
    )
    parser.add_argument(
        "-M",
        "--memory",
        default=-1,
        type=memory_t,
        help="The max allowed memory in bytes.",
    )

    parser.add_argument(
        "prog", nargs=argparse.REMAINDER, help="The program to run and monitor"
    )

    return parser.parse_args()


def get_memory_usage():
    try:
        p = psutil.Process()
        children = p.children(recursive=True)
        memory = 0
        for child in children:
            memory += child.memory_info().rss
    except psutil.NoSuchProcess:
        memory = 0
    return memory


def terminate(signum=None, frame=None):
    p = psutil.Process()
    children = p.children(recursive=True)
    for child in children:
        child.terminate()


def dispatch(prog, max_memory=-1, timeout=-1, log_period=2):
    logger = logging.getLogger("resmonitor")
    proc = sp.Popen(prog)

    start_t = time.time()
    last_log_t = float("-inf")
    try:
        while proc.poll() is None:
            time.sleep(0.01)
            now_t = time.time()
            duration_t = now_t - start_t
            mem_usage = get_memory_usage()
            if now_t - last_log_t > log_period:
                logger.info(
                    "Duration: %.3fs, MemUsage: %s", duration_t, memory_repr(mem_usage)
                )
                last_log_t = now_t
            if max_memory >= 0 and mem_usage >= max_memory:
                logger.info(
                    "Duration: %.3fs, MemUsage: %s", duration_t, memory_repr(mem_usage)
                )
                logger.error(
                    "Out of Memory (terminating process): %s > %s",
                    memory_repr(mem_usage),
                    memory_repr(mem_usage),
                )
                terminate()
                break
            if timeout >= 0 and duration_t >= timeout:
                logger.info(
                    "Duration: %.3fs, MemUsage: %s", duration_t, memory_repr(mem_usage)
                )
                logger.error(
                    "Timeout (terminating process): %.2f > %.2f", duration_t, timeout
                )
                terminate()
                break
        else:
            logger.info(
                "Duration: %.3fs, MemUsage: %s", duration_t, memory_repr(mem_usage)
            )
            logger.info("Process finished successfully.")
            terminate()
    except KeyboardInterrupt:
        logger.info("Duration: %.3fs, MemUsage: %s", duration_t, memory_repr(mem_usage))
        logger.error("Received keyboard interupt (terminating process)")
        terminate()
    return proc.wait()


def main(args):
    signal.signal(signal.SIGTERM, terminate)

    logger = logging.getLogger("resmonitor")

    if args.debug:
        logger.setLevel(logging.DEBUG)
        log_period = 1  # seconds
    elif args.verbose:
        logger.setLevel(logging.INFO)
        log_period = 2  # seconds
    elif args.quiet:
        logger.setLevel(logging.ERROR)
        log_period = float("inf")
    else:
        logger.setLevel(logging.INFO)
        log_period = 5  # seconds

    formatter = logging.Formatter(f"%(levelname)-8s %(asctime)s (%(name)s) %(message)s")

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return dispatch(
        args.prog, max_memory=args.memory, timeout=args.time, log_period=log_period
    )


if __name__ == "__main__":
    main(_parse_args())
