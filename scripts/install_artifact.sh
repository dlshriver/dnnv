#!/bin/bash

use_gurobi=1

print_usage() {
    echo "Usage: run.sh [--no-gurobi]"
    echo ""
    echo "Available options:"
    echo "  -h, --help       show this help message and exit"
    echo "  --no-gurobi      do not run verifiers that require gurobi"
}

while [ -n "$1" ]; do # while loop starts
    case "$1" in
    -h | --help)
        print_usage
        exit 0
        ;;
    --no-gurobi)
        use_gurobi=0
        ;;
    *)
        echo "Option $1 not recognized"
        print_usage
        exit 1
        ;;
    esac
    shift
done

sudo apt update
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install software-properties-common build-essential python3.7 python3.7-dev python3.7-venv cmake wget git liblapack-dev openssl libssl-dev valgrind libtool libboost-all-dev libglpk-dev qt5-qmake libltdl-dev protobuf-compiler

mkdir artifact
cd artifact

git clone https://github.com/dlshriver/DNNV.git
cd DNNV
. .env.d/openenv.sh

./manage.sh install reluplex
./manage.sh install planet
./manage.sh install neurify
./manage.sh install mipverify
./manage.sh install eran
./manage.sh install plnn
./manage.sh install marabou
./manage.sh install nnenum
./manage.sh install verinet

cd ..
wget "https://drive.google.com/u/1/uc?id=1RJDq4jsiteEE2RYRo3Nvmm7E8scTxPQm&export=download" -O cav2021_artifact.tar.gz
tar xzf cav2021_artifact.tar.gz

echo ". $(pwd)/DNNV/.env.d/openenv.sh" >> .bashrc
