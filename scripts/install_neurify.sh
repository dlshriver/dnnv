#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)}

cd $PROJECT_DIR/bin
wget https://github.com/Kitware/CMake/releases/download/v3.17.1/cmake-3.17.1-Linux-x86_64.sh
chmod u+x cmake-3.17.1-Linux-x86_64.sh
yes | ./cmake-3.17.1-Linux-x86_64.sh
cp -r cmake-3.17.1-Linux-x86_64/bin/* .
cp -r cmake-3.17.1-Linux-x86_64/share/* $PROJECT_DIR/share/

cd $PROJECT_DIR/lib
wget http://github.com/xianyi/OpenBLAS/archive/v0.3.6.tar.gz
tar -xzf v0.3.6.tar.gz
cd OpenBLAS-0.3.6
make
make PREFIX=$PROJECT_DIR install
cd ..
rm v0.3.6.tar.gz

cd $PROJECT_DIR
./scripts/install_lpsolve.sh

cd $PROJECT_DIR
./scripts/install_suitesparse.sh

cd $PROJECT_DIR/bin
git clone https://github.com/dlshriver/Neurify.git
cd Neurify
git checkout general
git pull
cd generic
make OPENBLAS_HOME=$PROJECT_DIR
cp src/neurify ../../
