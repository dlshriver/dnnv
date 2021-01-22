#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)}
source $PROJECT_DIR/scripts/install_common.sh
ensure_cmake

pip install onnxruntime

cd $PROJECT_DIR/bin
git clone https://github.com/NeuralNetworkVerification/Marabou
cd Marabou
git checkout 8bea650
mkdir build 
cd build
cmake ..
cmake --build .
cp Marabou $PROJECT_DIR/bin/marabou

cp $PROJECT_DIR/tools/verifier_runners/marabou.py $PROJECT_DIR/bin/marabou.py
