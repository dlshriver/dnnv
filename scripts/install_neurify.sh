#!/usr/bin/env bash

PROJECT_DIR=$(cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..; pwd)

cd $PROJECT_DIR/lib
wget http://github.com/xianyi/OpenBLAS/archive/v0.3.6.tar.gz
tar -xzf v0.3.6.tar.gz
cd OpenBLAS-0.3.6
make
make PREFIX=$PROJECT_DIR install
cd ..
rm v0.3.6.tar.gz

cd $PROJECT_DIR/lib
wget https://downloads.sourceforge.net/project/lpsolve/lpsolve/5.5.2.5/lp_solve_5.5.2.5_dev_ux64.tar.gz
tar -xzf lp_solve_5.5.2.5_dev_ux64.tar.gz
rm lp_solve_5.5.2.5_dev_ux64.tar.gz
mkdir $PROJECT_DIR/include/lpsolve/
cp lp_*.h $PROJECT_DIR/include/lpsolve/

cd $PROJECT_DIR/lib
wget https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v5.6.0.tar.gz
tar -xzf v5.6.0.tar.gz
cd SuiteSparse-5.6.0
make
cp lib/* $PROJECT_DIR/lib
rm v5.6.0.tar.gz

cd $PROJECT_DIR/bin
git clone https://github.com/dlshriver/Neurify.git
cd Neurify
git checkout general
git pull
cd generic
make OPENBLAS_HOME=$PROJECT_DIR
cp src/neurify ../../
