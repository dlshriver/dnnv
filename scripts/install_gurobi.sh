#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..; pwd)}

cd $PROJECT_DIR/bin
if [ -e gurobi810 ]
then
    echo "Gurobi is already installed"
    exit
fi

wget https://packages.gurobi.com/8.1/gurobi8.1.0_linux64.tar.gz
tar -xvf gurobi8.1.0_linux64.tar.gz
cd gurobi810/linux64
python3 setup.py install
cd ../../
rm gurobi8.1.0_linux64.tar.gz

echo "Gurobi requires a license, which can be obtained from gurobi.com"
