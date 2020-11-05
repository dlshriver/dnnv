#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)}
source $PROJECT_DIR/scripts/install_common.sh

ensure_cmake
ensure_openblas
ensure_lpsolve
ensure_suitesparse

cd $PROJECT_DIR/bin
git clone https://github.com/dlshriver/Neurify.git
cd Neurify
git checkout 90de94a
git pull
cd generic
make OPENBLAS_HOME=$PROJECT_DIR
cp src/neurify ../../
