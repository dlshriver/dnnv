#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)}
source $PROJECT_DIR/scripts/install_common.sh

cd $PROJECT_DIR/lib
git clone https://github.com/stanleybak/nnenum.git
cd nnenum
git checkout 0986d21
git pull

cp $PROJECT_DIR/tools/verifier_runners/nnenum.py $PROJECT_DIR/bin/nnenum.py

pip install "threadpoolctl>=2.1,<2.2"
pip install "swiglpk>=4.65,<4.66"
pip install "foolbox>=2.4,<2.5"
pip install "skl2onnx>=1.7,<1.8"
pip install "onnxruntime"
