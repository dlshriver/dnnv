#!/usr/bin/env bash

cd bin
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
