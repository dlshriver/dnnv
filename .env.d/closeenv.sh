#!/bin/bash

if [ -n "${VIRTUAL_ENV:x}" ]; then
    deactivate
fi

if [ -n "${PROJECT_ENV:x}" ]; then
    clear_vars
fi
