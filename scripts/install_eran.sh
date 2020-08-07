#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)}
source $PROJECT_DIR/scripts/install_common.sh

ensure_m4
ensure_gmp
ensure_mpfr
ensure_cddlib
ensure_elina
ensure_gurobi

pip install \
    --global-option=build_ext --global-option="-I$PROJECT_DIR/lib/gmp-6.1.2" \
    --global-option=build_ext --global-option="-L$PROJECT_DIR/lib/gmp-6.1.2/.libs" \
    pycddlib

cd $PROJECT_DIR/lib
git clone https://github.com/eth-sri/eran.git
cd eran
git pull
git checkout 0bbd864
