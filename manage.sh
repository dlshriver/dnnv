#!/bin/bash

mkdir -p bin
mkdir -p include
mkdir -p lib
mkdir -p share

if [ "$1" == "init" ]; then
    . .env.d/initenv.sh
fi

if [ "$1" == "update" ]; then
    . .env.d/openenv.sh
    python -m pip install --upgrade pip setuptools flit
    flit install -s
fi

if [ "$1" == "install" ]; then
    . .env.d/openenv.sh
    shift
    for pkg in "$@"; do
        if [ "$pkg" == "bab" ]; then
            echo "Installing PLNN/BAB..."
            ./scripts/install_plnn.sh
        elif [ "$pkg" == "eran" ]; then
            echo "Installing ERAN..."
            ./scripts/install_eran.sh
        elif [ "$pkg" == "mipverify" ]; then
            echo "Installing MIPVerify..."
            ./scripts/install_mipverify.sh
        elif [ "$pkg" == "neurify" ]; then
            echo "Installing Neurify..."
            ./scripts/install_neurify.sh
        elif [ "$pkg" == "nnenum" ]; then
            echo "Installing nnenum..."
            ./scripts/install_nnenum.sh
        elif [ "$pkg" == "planet" ]; then
            echo "Installing Planet..."
            ./scripts/install_planet.sh
        elif [ "$pkg" == "plnn" ]; then
            echo "Installing PLNN..."
            ./scripts/install_plnn.sh
        elif [ "$pkg" == "reluplex" ]; then
            echo "Installing Reluplex..."
            ./scripts/install_reluplex.sh
        elif [ "$pkg" == "marabou" ]; then
            echo "Installing Marabou..."
            ./scripts/install_marabou.sh
        elif [ "$pkg" == "verinet" ]; then
            echo "Installing VeriNet..."
            ./scripts/install_verinet.sh
        else
            echo "Unknown package: $pkg"
        fi
    done
fi
