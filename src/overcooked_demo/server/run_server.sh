#!/bin/bash

# This script runs the Overcooked server using uv

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Please install it first using 'pip install uv'"
    exit 1
fi

# Create a venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies if needed
if [ ! -f ".venv/.initialized" ]; then
    # Install dependencies from pyproject.toml
    echo "Installing dependencies from pyproject.toml..."
    uv pip install -e .
    
    # Also install root package
    echo "Ensuring overcooked_ai is installed..."
    ROOT_DIR="$(realpath "$(dirname "$0")/../..")"
    uv pip install -e "$ROOT_DIR"
    
    # Try to install tensorflow with a fallback
    echo "Attempting to install tensorflow..."
    if ! uv pip install -e ".[tensorflow]" 2>/dev/null; then
        echo "Could not install latest tensorflow. Trying compatible version..."
        # Try a version known to work on many platforms
        if ! uv pip install "tensorflow>=2.8.0" 2>/dev/null; then
            echo "Warning: Could not install tensorflow. The server will run with limited functionality."
        fi
    fi
    
    # Mark as initialized
    touch .venv/.initialized
fi

# Run the server using the Python interpreter from the venv
echo "Starting server..."
.venv/bin/python -m dev_helper app.py "$@" 