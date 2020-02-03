#!/bin/bash

PROJECT_DIR=$(cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..; pwd)
mkdir -p bin
mkdir -p include
mkdir -p lib

if [ "$1" == "init" ]
then
    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip setuptools
    while read req || [ -n "$req" ]
    do
        echo "pip install $req"
        pip install $req
    done < requirements.txt
fi

if [ "$1" == "update" ]
then
    if [ ! -d .venv ]
    then
        echo "Environment does not exist. Try:" >&2
        echo " ./manage.sh init"
        exit 1
    fi
    . .venv/bin/activate
    python -m pip install --upgrade pip setuptools
    while read req || [ -n "$req" ]
    do
        echo "pip install --upgrade $req"
        pip install --upgrade $req
    done < requirements.txt
fi

if [ "$1" == "install" ]
then
    shift
    for pkg in "$@"
    do
        if [ "$pkg" == "bab" ]
        then
            echo "Installing PLNN/BAB..."
            ./scripts/install_plnn.sh
        elif [ "$pkg" == "eran" ]
        then
            echo "Installing ERAN..."
            ./scripts/install_eran.sh
        elif [ "$pkg" == "gurobi" ]
        then
            echo "Installing Gurobi..."
            ./scripts/install_gurobi.sh
        elif [ "$pkg" == "mipverify" ]
        then
            echo "Installing MIPVerify..."
            ./scripts/install_mipverify.sh
        elif [ "$pkg" == "neurify" ]
        then
            echo "Installing Neurify..."
            ./scripts/install_neurify.sh
        elif [ "$pkg" == "planet" ]
        then
            echo "Installing Planet..."
            ./scripts/install_planet.sh
        elif [ "$pkg" == "plnn" ]
        then
            echo "Installing PLNN..."
            ./scripts/install_plnn.sh
        elif [ "$pkg" == "reluplex" ]
        then
            echo "Installing Reluplex..."
            ./scripts/install_reluplex.sh
        else
            echo "Unknown package: $pkg"
        fi
    done
fi