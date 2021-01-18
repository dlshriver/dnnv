#!/bin/bash

if [ -n "$PROJECT_DIR" ]; then
    echo "Closing open env: $PROJECT_ENV ($PROJECT_DIR)"
    . $PROJECT_DIR/.env.d/closeenv.sh
fi

envdir=$(dirname "${BASH_SOURCE[0]}")
currentproject=$PROJECT_ENV

. $envdir/env.sh

if [ "$currentproject" == "$PROJECT_ENV" ]; then
    . $envdir/closeenv.sh
    . $envdir/env.sh
fi
unset currentproject
unset envdir

append_path $PROJECT_DIR/bin/ PATH
append_path $PROJECT_DIR/lib/ LD_LIBRARY_PATH
append_path $PROJECT_DIR PYTHONPATH

if [ -e ./.venv/bin/activate ]; then
    . $PROJECT_DIR/.venv/bin/activate
else
    echo "Environment does not exist. Initializing..."
    $PROJECT_DIR/.env.d/initenv.sh
    . $PROJECT_DIR/.venv/bin/activate
fi

# gurobi paths
set_var GUROBI_HOME $PROJECT_DIR/bin/gurobi902/linux64
append_path $GUROBI_HOME/bin PATH
append_path $GUROBI_HOME/lib LD_LIBRARY_PATH

# eran paths
append_path $PROJECT_DIR/lib/eran/tf_verify PYTHONPATH
append_path $PROJECT_DIR/lib/ELINA/python_interface PYTHONPATH
append_path $PROJECT_DIR/lib/eran/ELINA/python_interface PYTHONPATH

# julia paths
append_path $PROJECT_DIR/bin/julia-1.0.4/bin PATH

# nnenum paths
append_path $PROJECT_DIR/lib/nnenum/nnenum PYTHONPATH

# marabou paths
append_path $PROJECT_DIR/bin/Marabou PYTHONPATH
