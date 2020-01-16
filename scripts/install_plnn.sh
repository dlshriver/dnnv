#!/usr/bin/env bash

PROJECT_DIR=$(cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..; pwd)
cd lib

git clone https://github.com/oval-group/PLNN-verification.git
cd PLNN-verification
git checkout newVersion
git pull
python setup.py install
rm -rf convex_adversarial
git clone https://github.com/locuslab/convex_adversarial
cd convex_adversarial
python setup.py install

cd $PROJECT_DIR
./scripts/install_gurobi.sh
