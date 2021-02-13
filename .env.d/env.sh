#!/bin/bash

export PROJECT_ENV="DNNV"
export PROJECT_DIR=$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)

PROJECT_ENVVARS="__${PROJECT_ENV}_VARS"

function dnnv_install() {
    $PROJECT_DIR/manage.sh install $1
}

function set_var() {
    var=$1
    val=$2
    old_var="__${PROJECT_ENV}_OLD_${var}"
    old_val=${!var}
    result=$(echo ${!PROJECT_ENVVARS} | egrep ":$var:|^$var:" || echo "$")
    if [ "$result" == "$" ]; then
        export $PROJECT_ENVVARS=$var:${!PROJECT_ENVVARS}
        export $old_var=$old_val
    else
        echo "WARNING: variable $var set more than once"
    fi
    export $var=$val
}

function append_path() {
    var=${2:-PATH}
    val=$1
    old_var="__${PROJECT_ENV}_OLD_${var}"
    old_val=${!var}
    result=$(echo ${!PROJECT_ENVVARS} | egrep ":$var:|^$var:" || echo "$")
    if [ "$result" == "$" ]; then
        export "$PROJECT_ENVVARS=$var:${!PROJECT_ENVVARS}"
        export "$old_var=$old_val"
    fi
    export "$var=$val:$old_val"
}

function clear_vars() {
    for var in $(echo ${!PROJECT_ENVVARS} | tr ":" "\n"); do
        old_var="__${PROJECT_ENV}_OLD_${var}"
        export "$var=${!old_var}"
        unset $old_var
    done
    unset $PROJECT_ENVVARS
    unset PROJECT_ENV
    unset PROJECT_DIR
    unset PROJECT_ENVVARS
    unset append_path
    unset clear_vars
    unset set_var
    unset dnnv_install
}
