#!/bin/bash

set -e

dnnv_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python=python3

print_usage() {
    echo "Usage: install.sh [-p exe | --python exe]"
    echo ""
    echo "Available options:"
    echo "  -h, --help       show this help message and exit"
    echo "  -p, --python exe"
    echo "                   the python executable to use"
}

while [ -n "$1" ]; do # while loop starts
    case "$1" in
    -h | --help)
        print_usage
        exit 0
        ;;
    -p | --python)
        python=$2
        shift
        ;;
    *)
        echo "Option $1 not recognized"
        print_usage
        exit 1
        ;;
    esac
    shift
done

if ! command -v $python &>/dev/null; then
    echo "$python could not be found"
    exit 1
fi

python_major_version=$($python -c "import sys; print(sys.version_info.major)")
python_minor_version=$($python -c "import sys; print(sys.version_info.minor)")

if [ $python_major_version -ne 3 ]; then
    echo "DNNV only supports Python 3"
    exit 1
elif [ $python_minor_version -lt 7 ]; then
    echo "DNNV only supports Python 3.7+"
    exit 1
fi

$python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip setuptools flit

flit install -s
