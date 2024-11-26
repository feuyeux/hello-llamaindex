#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
cd "$SCRIPT_DIR" || exit

python -m venv llamaindex_env

if [[ "$(uname -s)" == *"Linux"* ]] || [[ "$(uname -s)" == *"Darwin"* ]]; then
    echo "install on WSL, Linux or macOS"
    source "$SCRIPT_DIR/llamaindex_env/bin/activate"
    which python
    pip install --upgrade pip
elif [[ "$(uname -s)" == *"CYGWIN"* ]] || [[ "$(uname -s)" == *"MSYS"* ]] || [[ "$(uname -s)" == *"MINGW"* ]]; then
    echo "install on Cygwin or MSYS2"
    . "llamaindex_env/Scripts/activate"
    which python
    python.exe -m pip install --upgrade pip
else
    echo "Unsupported OS"
    exit 1
fi
pip install -r requirements.txt
