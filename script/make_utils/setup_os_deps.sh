#!/usr/bin/env bash

# From https://stackoverflow.com/a/69860299
isDocker(){
    local cgroup=/proc/1/cgroup
    test -f $cgroup && [[ "$(<$cgroup)" = *:cpuset:/docker/* ]]
}

isDockerBuildkit(){
    local cgroup=/proc/1/cgroup
    test -f $cgroup && [[ "$(<$cgroup)" = *:cpuset:/docker/buildkit/* ]]
}

isDockerContainer(){
    [[ -e /.dockerenv ]]
}

LINUX_INSTALL_PYTHON=0

while [ -n "$1" ]
do
   case "$1" in
        "--linux-install-python" )
            LINUX_INSTALL_PYTHON=1
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

OS_NAME=$(uname)

if [[ "${OS_NAME}" == "Linux" ]]; then
    # Docker build
    if isDockerBuildkit || (isDocker && ! isDockerContainer); then
        CLEAR_APT_LISTS="rm -rf /var/lib/apt/lists/* &&"
        SUDO_BIN=""
    else
        CLEAR_APT_LISTS=""
        SUDO_BIN="$(command -v sudo)"
        if [[ "${SUDO_BIN}" != "" ]]; then
            SUDO_BIN="${SUDO_BIN} "
        fi
    fi

    PYTHON_PACKAGES=
    if [[ "${LINUX_INSTALL_PYTHON}" == "1" ]]; then
        PYTHON_PACKAGES="python3-pip \
        python3.8 \
        python3.8-dev \
        python3.8-tk \
        python3.8-venv \
        python-is-python3 \
        "
    fi

    SETUP_CMD="${SUDO_BIN:+$SUDO_BIN}apt-get update && apt-get upgrade --no-install-recommends -y && \
    ${SUDO_BIN:+$SUDO_BIN}apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    ${PYTHON_PACKAGES:+$PYTHON_PACKAGES} \
    git \
    graphviz* \
    jq \
    make \
    pandoc \
    shellcheck && \
    ${CLEAR_APT_LISTS:+$CLEAR_APT_LISTS} \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry"
    eval "${SETUP_CMD}"
elif [[ "${OS_NAME}" == "Darwin" ]]; then
    brew install curl git graphviz jq make pandoc shellcheck
    python3 -m pip install -U pip
    python3 -m pip install poetry

    echo "Make is currently installed as gmake"
    echo 'If you need to use it as "make", you can add a "gnubin" directory to your PATH from your bashrc like:'
    # shellcheck disable=SC2016
    echo 'PATH="/usr/local/opt/make/libexec/gnubin:$PATH"'
else
    echo "Unknown OS"
    exit 1
fi
