
"""
Config file

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import logging


class CONFIG:

    # Logging:
    USE_LOGGER = False
    LOGS_LEVEL = logging.INFO
    LOGS_LEVEL_VERINET = logging.INFO
    LOGS_LEVEL_VERIFIER = logging.INFO
    LOGS_TERMINAL = True

    # General:
    USE_PROGRESS_BARS = False
    PRECISION = 32  # 32 or 64, applies to SIP only.

    # Multiprocessing:
    CHILDPROC_TYPE = "forkserver"
    QUEUE_DEPTH = 1
    MAX_CHILDREN_SUSPEND_TIME = 600
    MAX_ACCEPTED_MEMORY_INCREASE = 20
    USE_ONE_SHOT_ATTEMPT = True
    USE_PRE_PROCESSING_ATTACK = True  # Only used if 'use_one_shot_attempt' is True

    # Profiling:
    PROFILE_WORKER = False

    # Verifier:
    MAX_QUEUED_BRANCHES = None

    # RSIP
    MAX_ESTIMATED_MEM_USAGE = 64 * 10 ** 9
    OPTIMISED_RELU_RELAXATION_MAX_BOUNDS_MULTIPLIER = 3

    # Pre-processing:
    USE_SSIP = False
    STORE_SSIP_BOUNDS = False

    # Split domains:
    INPUT_NODE_SPLIT = True
    HIDDEN_NODE_SPLIT = True
    INDIRECT_HIDDEN_MULTIPLIER = 0.75
    INDIRECT_INPUT_MULTIPLIER = 0.75

    # LP-Problem:
    USE_BIAS_SEPARATED_CONSTRAINTS = True
    PERFORM_LP_MAXIMISATION = False
    USE_OPTIMISED_RELAXATION_CONSTRAINTS = True
    USE_OPTIMISED_RELAXATION_SPLIT_HEURISTIC = True
    USE_SIMPLE_LP = True
    NUM_ITER_OPTIMISED_RELAXATIONS = 3
    USE_LP_PRESOLVE = 0  # 0 for off, 1 for on. The problem is usually solved faster without Presolve.

    # Gradient Descent:
    GRADIENT_DESCENT_INTERVAL = 1
    GRADIENT_DESCENT_MAX_ITERS = 5
    GRADIENT_DESCENT_STEP = 0.1
    GRADIENT_DESCENT_MIN_LOSS_CHANGE = 1e-2


logging.basicConfig(level=CONFIG.LOGS_LEVEL_VERINET)
