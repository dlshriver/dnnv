#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)}
source $PROJECT_DIR/scripts/install_common.sh

ensure_gurobi

cp $PROJECT_DIR/tools/verifier_runners/verinet.py $PROJECT_DIR/bin/verinet.py

pip install "numba>=0.50,<0.60"
