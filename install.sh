#!/bin/bash
set -e  # Exit on error
set -u  # Exit on undefined variables

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

cleanup() {
    if [ -d "llamaindex_env" ]; then
        log "Cleaning up virtual environment..."
        rm -rf llamaindex_env
    fi
}

check_command() {
    command -v "$1" >/dev/null 2>&1 || error "$1 is required but not installed."
}

# Check required commands
check_command python
check_command pip

# Set up error handling
trap cleanup ERR

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
cd "$SCRIPT_DIR" || error "Failed to change to script directory"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    error "requirements.txt not found in $SCRIPT_DIR"
fi

log "Creating Python virtual environment..."
python -m venv llamaindex_env || error "Failed to create virtual environment"

if [[ "$(uname -s)" == *"Linux"* ]] || [[ "$(uname -s)" == *"Darwin"* ]]; then
    log "Installing on WSL, Linux or macOS"
    source "$SCRIPT_DIR/llamaindex_env/bin/activate" || error "Failed to activate virtual environment"
    which python || error "Python not found after activation"
    pip install --upgrade pip || error "Failed to upgrade pip"
elif [[ "$(uname -s)" == *"CYGWIN"* ]] || [[ "$(uname -s)" == *"MSYS"* ]] || [[ "$(uname -s)" == *"MINGW"* ]]; then
    log "Installing on Cygwin or MSYS2"
    . "llamaindex_env/Scripts/activate" || error "Failed to activate virtual environment"
    which python || error "Python not found after activation"
    python.exe -m pip install --upgrade pip || error "Failed to upgrade pip"
else
    error "Unsupported operating system"
fi

log "Installing required packages..."
pip install -r requirements.txt || error "Failed to install required packages"

log "Installation completed successfully!"
