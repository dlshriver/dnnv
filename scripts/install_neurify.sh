#!/usr/bin/env bash

PROJECT_DIR=$(cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..; pwd)

cd lib
wget http://github.com/xianyi/OpenBLAS/archive/v0.3.6.tar.gz
tar -xzf v0.3.6.tar.gz
cd OpenBLAS-0.3.6
make
make PREFIX=$PROJECT_DIR install
cd ..
rm v0.3.6.tar.gz
cd ..

cd bin
git clone https://github.com/dlshriver/Neurify.git
cd Neurify
git checkout general
git pull
cd generic
make OPENBLAS_HOME=$PROJECT_DIR
cp src/neurify ../../
cp src/lp_dev/liblpsolve55.so ../../../lib/
