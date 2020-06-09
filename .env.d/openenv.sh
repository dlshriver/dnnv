#!/bin/bash

if [ -z "${DISPLAY:x}" ]
then
    . ~/.bashrc
fi

if [ -e ./.venv/bin/activate ]
then
    . .venv/bin/activate
fi

export ENV_OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export ENV_OLD_PATH=$PATH
export ENV_OLD_PYTHONPATH=$PYTHONPATH

PROJECT_DIR=$(cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..; pwd)

export PATH=$PROJECT_DIR/bin/:$PATH
export LD_LIBRARY_PATH=$PROJECT_DIR/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# gurobi paths
export GUROBI_HOME=$PROJECT_DIR/bin/gurobi810/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH

# eran paths
export PYTHONPATH=$PROJECT_DIR/lib/eran/tf_verify:$PYTHONPATH
export PYTHONPATH=$PROJECT_DIR/lib/eran/ELINA/python_interface:$PYTHONPATH

# julia paths
export PATH=$PROJECT_DIR/bin/julia-1.0.4/bin:$PATH
