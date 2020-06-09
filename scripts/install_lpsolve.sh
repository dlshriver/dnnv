#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)}

cd $PROJECT_DIR/include
if [ -e lpsolve ]; then
    echo "lpsolve is already installed"
    exit
fi

cd $PROJECT_DIR/lib
wget https://downloads.sourceforge.net/project/lpsolve/lpsolve/5.5.2.5/lp_solve_5.5.2.5_dev_ux64.tar.gz
tar -xzf lp_solve_5.5.2.5_dev_ux64.tar.gz
rm lp_solve_5.5.2.5_dev_ux64.tar.gz
mkdir $PROJECT_DIR/include/lpsolve/
cp lp_*.h $PROJECT_DIR/include/lpsolve/
