#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)}

cd $PROJECT_DIR/lib
if [ -e libsuitesparseconfig.so ]; then
    echo "SuiteSparse is already installed"
    exit
fi

cd $PROJECT_DIR/lib
wget https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v5.6.0.tar.gz
tar -xzf v5.6.0.tar.gz
cd SuiteSparse-5.6.0
make library
cp lib/* $PROJECT_DIR/lib
cp */Lib/*.a $PROJECT_DIR/lib
cp SuiteSparse_config/libsuitesparseconfig.a $PROJECT_DIR/lib
rm v5.6.0.tar.gz
