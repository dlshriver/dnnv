#!/bin/bash

if [ -n "${VIRTUAL_ENV:x}" ]; then
    . $PROJECT_DIR/.venv/bin/activate
    deactivate
fi

if [ -n "${PROJECT_ENV:x}" ]; then
    envdir=$(dirname "${BASH_SOURCE[0]}")
    . $envdir/env.sh
    clear_vars
fi
