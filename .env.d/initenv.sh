#!/bin/bash

PROJECT_ENV="DNNV"
PROJECT_DIR=$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)

echo "Initializing $PROJECT_ENV in directory $PROJECT_DIR"
mkdir -p $PROJECT_DIR/bin
mkdir -p $PROJECT_DIR/include
mkdir -p $PROJECT_DIR/lib
mkdir -p $PROJECT_DIR/share

python3.7 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools flit
flit install -s
