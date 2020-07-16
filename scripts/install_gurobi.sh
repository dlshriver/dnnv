#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..; pwd)}

cd $PROJECT_DIR/bin
if [ -e gurobi902 ]
then
    echo "Gurobi is already installed"
    exit
fi

wget https://packages.gurobi.com/9.0/gurobi9.0.2_linux64.tar.gz
tar -xvf gurobi9.0.2_linux64.tar.gz
cd gurobi902/linux64
cp lib/libgurobi90.so $PROJECT_DIR/lib
python3 setup.py install
cd ../../
rm gurobi9.0.2_linux64.tar.gz

echo "Gurobi requires a license, which can be obtained from gurobi.com"
