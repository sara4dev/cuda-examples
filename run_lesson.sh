#!/bin/bash
# Wrapper script to set up CUDA library paths

# Find Python - check if in venv, otherwise use system python
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
    VENV_PATH="$VIRTUAL_ENV"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    VENV_PATH="$(dirname "$(dirname "$(which python)")")"
else
    PYTHON_CMD="python3"
    VENV_PATH="$(dirname "$(dirname "$(which python3)")")"
fi

CUDA_LIB_PATH="${VENV_PATH}/lib/python3.12/site-packages/nvidia"

export LD_LIBRARY_PATH="${CUDA_LIB_PATH}/cuda_nvrtc/lib:${CUDA_LIB_PATH}/cuda_runtime/lib:${LD_LIBRARY_PATH}"

# Run the Python script
exec "$PYTHON_CMD" "$@"


