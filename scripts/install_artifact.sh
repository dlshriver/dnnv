#!/bin/bash

set -e

sudo apt update
sudo apt install -y software-properties-common build-essential
sudo apt update
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.7 python3.7-dev python3.7-venv 
sudo apt install -y cmake wget git liblapack-dev openssl libssl-dev valgrind libtool libboost1.71-all-dev libglpk-dev qt5-qmake libltdl-dev protobuf-compiler

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

echo ". $(pwd)/DNNV/.env.d/openenv.sh" >> ~/.bashrc
